"""Run a preregistered experiment plan continuously and expose boundary events.

The supervisor never interprets experiment metrics. It only executes immutable
stage commands in order, records atomic status, and prevents duplicate runners.
Codex can block on ``--wait-after-revision`` without reading training logs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def validate_plan(plan: dict) -> None:
    if plan.get("schema_version") != "1.0" or not SAFE_ID.fullmatch(str(plan.get("plan_id", ""))):
        raise ValueError("invalid plan identity")
    stages = plan.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("plan requires at least one stage")
    seen = set()
    for stage in stages:
        stage_id = str(stage.get("id", ""))
        command = stage.get("command")
        if not SAFE_ID.fullmatch(stage_id) or stage_id in seen:
            raise ValueError("stage ids must be unique safe names")
        if not isinstance(command, list) or not command or not all(isinstance(value, str) and value for value in command):
            raise ValueError(f"stage {stage_id} requires an argv command")
        if stage.get("cwd") is not None and not Path(stage["cwd"]).is_absolute():
            raise ValueError("stage cwd must be absolute")
        if not isinstance(stage.get("env", {}), dict):
            raise ValueError("stage env must be an object")
        seen.add(stage_id)
    if plan.get("terminal_state") not in ("AWAITING_GATE_REVIEW", "READY_FOR_CLOSURE"):
        raise ValueError("plan needs an explicit terminal state")


def paths(plan_path: Path, plan: dict) -> tuple[Path, Path, Path, Path]:
    control_root = Path(plan.get("control_root", plan_path.parent / "control")).resolve()
    directory = control_root / plan["plan_id"]
    return directory, directory / "status.json", directory / "events.jsonl", directory / "supervisor.lock"


def append_event(events_path: Path, status: dict) -> None:
    event = {key: status.get(key) for key in (
        "revision", "time_utc", "state", "current_stage", "completed_stage_ids", "last_exit_code", "message"
    )}
    with events_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def publish(status_path: Path, events_path: Path, status: dict, **updates) -> None:
    status.update(updates)
    status["revision"] = int(status.get("revision", 0)) + 1
    status["time_utc"] = now()
    atomic_json(status_path, status)
    append_event(events_path, status)


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def acquire_lock(lock_path: Path) -> int:
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        try:
            pid = int(lock_path.read_text().strip())
        except Exception as error:
            raise RuntimeError("invalid supervisor lock") from error
        if process_exists(pid):
            raise RuntimeError(f"plan already has live supervisor pid={pid}")
        lock_path.unlink()
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor, f"{os.getpid()}\n".encode())
    os.fsync(descriptor)
    return descriptor


def run_plan(plan_path: Path) -> None:
    plan = read_json(plan_path)
    validate_plan(plan)
    plan_sha = canonical_hash(plan)
    directory, status_path, events_path, lock_path = paths(plan_path, plan)
    directory.mkdir(parents=True, exist_ok=True)
    descriptor = acquire_lock(lock_path)
    try:
        if status_path.exists():
            status = read_json(status_path)
            if status.get("plan_sha256") != plan_sha:
                raise RuntimeError("existing status belongs to a different immutable plan")
            if status.get("state") in (plan["terminal_state"], "PLAN_COMPLETE"):
                return
            if status.get("state") == "RUNNING":
                raise RuntimeError("unclean supervisor interruption requires manual stage audit")
        else:
            status = {
                "schema_version": "1.0", "plan_id": plan["plan_id"], "plan_sha256": plan_sha,
                "revision": 0, "state": "CREATED", "current_stage": None,
                "completed_stage_ids": [], "last_exit_code": None, "message": "plan registered",
                "supervisor_pid": os.getpid(), "plan_path": str(plan_path.resolve()),
            }
            publish(status_path, events_path, status)
        completed = list(status.get("completed_stage_ids", []))
        for stage in plan["stages"]:
            if stage["id"] in completed:
                continue
            log_path = directory / f"stage-{len(completed):02d}-{stage['id']}.log"
            publish(status_path, events_path, status, state="RUNNING", current_stage=stage["id"],
                    last_exit_code=None, message="stage started", supervisor_pid=os.getpid(), log=str(log_path))
            environment = os.environ.copy()
            environment.update({str(key): str(value) for key, value in stage.get("env", {}).items()})
            with log_path.open("ab", buffering=0) as stream:
                process = subprocess.Popen(stage["command"], cwd=stage.get("cwd"), env=environment,
                                           stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                publish(status_path, events_path, status, state="RUNNING", current_stage=stage["id"],
                        child_pid=process.pid, message="child launched")
                return_code = process.wait()
            if return_code:
                publish(status_path, events_path, status, state="FAILED", current_stage=stage["id"],
                        child_pid=None, last_exit_code=return_code, message="stage failed; no later stage launched")
                raise SystemExit(return_code)
            completed.append(stage["id"])
            publish(status_path, events_path, status, state="STAGE_COMPLETE", current_stage=stage["id"],
                    child_pid=None, completed_stage_ids=completed, last_exit_code=0, message="stage complete")
        publish(status_path, events_path, status, state=plan["terminal_state"], current_stage=None,
                child_pid=None, last_exit_code=0, message="all preregistered stages complete; gate review required")
    finally:
        os.close(descriptor)
        if lock_path.exists():
            lock_path.unlink()


def wait_event(plan_path: Path, revision: int, interval: float) -> None:
    plan = read_json(plan_path)
    validate_plan(plan)
    _, status_path, _, _ = paths(plan_path, plan)
    while True:
        if status_path.exists():
            status = read_json(status_path)
            if int(status.get("revision", 0)) > revision:
                print(json.dumps(status, sort_keys=True), flush=True)
                return
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--wait-after-revision", type=int)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    plan = read_json(args.plan)
    validate_plan(plan)
    _, status_path, _, _ = paths(args.plan, plan)
    if args.status:
        print(status_path.read_text() if status_path.exists() else json.dumps({"state": "NOT_STARTED"}))
    elif args.wait_after_revision is not None:
        wait_event(args.plan, args.wait_after_revision, args.poll_seconds)
    else:
        run_plan(args.plan)


if __name__ == "__main__":
    main()
