"""Read-only Codex quota fallback from this task's authenticated response metadata.

Never reads auth files, exports credentials, changes login, starts a model turn,
redeems credits, or operates AutoDL. Cache contains only quota fields/account hash.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".local/codex-usage/latest.json"


def account_tag(value):
    return hashlib.sha256(value.encode()).hexdigest() if value else None


def normalize(payload, source, expected_tag=None):
    buckets = payload.get("rateLimitsByLimitId") or {}
    bucket = buckets.get("codex") or payload.get("rateLimits")
    if not isinstance(bucket, dict):
        raise ValueError("quota_fields_missing")
    windows = {}
    for name in ("primary", "secondary"):
        window = bucket.get(name)
        if not isinstance(window, dict):
            windows[name] = None
            continue
        used = window.get("usedPercent")
        if isinstance(used, bool) or not isinstance(used, (float, int)) or not math.isfinite(used):
            raise ValueError("invalid_usage_percent")
        windows[name] = {"remaining_percent": max(0, min(100, 100-used)),
            "window_minutes": window.get("windowDurationMins"), "resets_at": window.get("resetsAt")}
    if not any(windows.values()):
        raise ValueError("quota_windows_missing")
    return {"observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source, "account_tag": expected_tag or account_tag(payload.get("accountId")),
        "bucket": bucket.get("limitId", "codex"), "windows": windows}


def decision(snapshot, now=None, estimated_next=0, reserve=3):
    now = now or datetime.now(timezone.utc)
    age = (now-datetime.fromisoformat(snapshot["observed_at_utc"])).total_seconds()
    values = [w["remaining_percent"] for w in snapshot["windows"].values() if w is not None]
    # Missing either window cannot silently be treated as unlimited/zero usage.
    complete = all(snapshot["windows"].get(k) is not None for k in ("primary", "secondary"))
    if not complete or age < 0 or age > 300:
        action = "UNKNOWN_DO_NOT_DISPATCH"
    elif min(values) < 10 or min(values) < estimated_next+reserve:
        action = "BACKUP_AND_CLOSE"
    else:
        action = "LIVE_BUDGET_AVAILABLE"
    return {**snapshot, "age_seconds": round(age, 1), "action": action,
        "estimated_next_percent": estimated_next, "shutdown_reserve_percent": reserve}


def save(snapshot, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(snapshot, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def read_session(root, thread_id):
    import uuid
    uuid.UUID(thread_id)
    latest = None
    for path in root.rglob("*"+thread_id+"*.jsonl"):
        # Current tasks can have segmented *_core-id files. Require actual
        # session_meta ID too; a different task's similarly named file is ignored.
        with path.open(encoding="utf-8") as stream:
            first = json.loads(next(stream))
            if first.get("type") != "session_meta" or first.get("payload", {}).get("id") != thread_id:
                continue
            for line in stream:
                try: record = json.loads(line)
                except json.JSONDecodeError: continue  # in-progress trailing write
                payload = record.get("payload", {})
                if record.get("type") != "event_msg" or payload.get("type") != "token_count": continue
                limits = payload.get("rate_limits")
                if not limits or limits.get("limit_id") != "codex": continue
                timestamp = record.get("timestamp")
                if latest and timestamp <= latest[0]: continue
                latest = timestamp, limits
    if not latest: raise ValueError("current_thread_quota_event_missing")
    timestamp, limits = latest
    bucket = {"limitId": limits["limit_id"]}
    for name in ("primary", "secondary"):
        window = limits.get(name)
        bucket[name] = ({"usedPercent": window.get("used_percent"),
            "windowDurationMins": window.get("window_minutes"), "resetsAt": window.get("resets_at")} if window else None)
    snapshot = normalize({"rateLimits": bucket}, "current_thread_response_event")
    snapshot["observed_at_utc"] = timestamp  # NEVER pretend an old event was just fetched
    snapshot["thread_id"] = thread_id
    snapshot["identity_scope"] = "authenticated_response_of_this_exact_thread"
    return snapshot


def cli_path(explicit=None):
    if explicit:
        path = Path(explicit).resolve()
        if not path.is_file(): raise ValueError("cli_not_found")
        return str(path)
    found = shutil.which("codex")
    if found: return found
    root = Path.home() / "AppData/Local/OpenAI/Codex/bin"
    candidates = list(root.glob("*/codex.exe"))
    if not candidates: raise ValueError("cli_not_found")
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def read_cli(executable, deadline_seconds, expected_tag):
    process = subprocess.Popen([executable, "app-server", "--stdio"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, encoding="utf-8", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    inbox = queue.Queue()
    def drain():
        for line in process.stdout:
            try: inbox.put(json.loads(line))
            except json.JSONDecodeError: continue
        inbox.put(None)
    threading.Thread(target=drain, daemon=True).start()
    deadline = time.monotonic()+deadline_seconds
    def send(message):
        process.stdin.write(json.dumps(message)+"\n"); process.stdin.flush()
    def response(identifier):
        while True:
            remaining = deadline-time.monotonic()
            if remaining <= 0: raise TimeoutError("rpc_deadline")
            message = inbox.get(timeout=remaining)
            if message is None: raise ValueError("app_server_exited")
            if message.get("id") != identifier: continue
            if "error" in message:
                # Raw errors may contain sensitive URLs; expose only integer RPC code.
                raise ValueError("rpc_error_"+str(message["error"].get("code", "unknown")))
            return message.get("result", {})
    try:
        send({"id": 1, "method": "initialize", "params": {"clientInfo": {
            "name": "vla_usage_reader", "title": "VLA quota read-only fallback", "version": "1.0"}}})
        response(1); send({"method": "initialized"})
        send({"id": 2, "method": "account/read", "params": {"refreshToken": False}})
        account = response(2).get("account") or {}
        tag = account_tag(account.get("id"))
        if account.get("type") != "chatgpt":
            raise ValueError("chatgpt_account_unavailable")
        if not tag:
            raise ValueError("cli_account_identity_unverified")
        if expected_tag and tag != expected_tag:
            raise ValueError("account_mismatch")
        if not expected_tag:
            raise ValueError("app_account_not_verified_record_primary_first")
        send({"id": 3, "method": "account/rateLimits/read"})
        return normalize(response(3), "official_app_server_stdio", tag)
    finally:
        process.stdin.close()
        try: process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.terminate()
            try: process.wait(timeout=1)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        process.stdout.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--record-tool-stdin", action="store_true")
    p.add_argument("--cli")
    p.add_argument("--thread-id", help="Preferred fallback: exact current task ID, no RPC or auth access")
    p.add_argument("--session-root", type=Path, default=Path.home()/".codex/sessions")
    p.add_argument("--timeout", type=float, default=18)
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--estimated-next-percent", type=float, default=0)
    p.add_argument("--reserve-percent", type=float, default=3)
    args = p.parse_args()
    if not 1 <= args.timeout <= 25 or args.reserve_percent < 3 or args.estimated_next_percent < 0:
        p.error("timeout must be 1..25s; reserve >=3%; next estimate >=0")
    import sys
    if args.record_tool_stdin:
        snapshot = normalize(json.load(sys.stdin), "codex_app_primary_tool")
        if not snapshot["account_tag"]: raise ValueError("primary_account_id_missing")
        save(snapshot, args.cache)
        print(json.dumps(decision(snapshot, estimated_next=args.estimated_next_percent,
                                  reserve=args.reserve_percent)))
        return 0
    cached = json.loads(args.cache.read_text()) if args.cache.exists() else None
    try:
        if not args.thread_id and not args.cli:
            raise ValueError("current_thread_id_required")
        snapshot = (read_session(args.session_root, args.thread_id) if args.thread_id else
                    read_cli(cli_path(args.cli), args.timeout, cached.get("account_tag") if cached else None))
        if decision(snapshot)["action"] == "UNKNOWN_DO_NOT_DISPATCH":
            raise ValueError("quota_event_stale_or_incomplete")
        save(snapshot, args.cache)
        result = decision(snapshot, estimated_next=args.estimated_next_percent, reserve=args.reserve_percent)
        print(json.dumps(result)); return 0
    except (ValueError, TimeoutError, queue.Empty, OSError, subprocess.SubprocessError) as error:
        # Cached values stay explicit historical observations, never fake fresh readings.
        print(json.dumps({"source": "fallback_failed", "error_category":
            str(error) if isinstance(error, ValueError) else type(error).__name__,
            "action": "UNKNOWN_DO_NOT_DISPATCH", "cached": ({**cached,
             "age_seconds": round((datetime.now(timezone.utc)-datetime.fromisoformat(cached["observed_at_utc"])).total_seconds(),1)} if cached else None)}))
        return 2


if __name__ == "__main__": raise SystemExit(main())
