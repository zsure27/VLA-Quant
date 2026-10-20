"""Quota decisions and native shutdown. No access to login secrets or platform boot."""
from __future__ import annotations

from datetime import datetime, timezone
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
from .common import digest, now, read_json, write_json
from .config import require


def quota_decision(observation, control, current=None):
    current = current or datetime.now(timezone.utc)
    require(observation.get("source") == "codex_api" and observation.get("window_minutes") == 300,
            "Use actual current five-hour primary API observation")
    age = (current - datetime.fromisoformat(observation["observed_at"].replace("Z", "+00:00"))).total_seconds()
    require(0 <= age <= control.get("quota_max_age_seconds", 1020), "Quota unavailable/stale; save and close")
    remaining, resets_at = observation["remaining_percent"], observation["resets_at"]
    require(type(remaining) in (int, float) and math.isfinite(remaining) and 0 <= remaining <= 100,
            "Invalid quota remaining")
    require(type(resets_at) in (int, float) and current.timestamp() < resets_at <= current.timestamp() + 18300,
            "No inferred refresh; need future actual resets_at")
    if remaining <= control["close_remaining_percent"]:
        return "CLOSE"
    if remaining <= control["stop_expansion_remaining_percent"]:
        return "STOP_EXPANSION"
    return "CONTINUE"


def native_shutdown(run_dir, expected_hostname, execute=False):
    """Preserve persistent files; request known AutoDL supervisor termination only."""
    require(os.name == "posix" and socket.gethostname() == expected_hostname, "Shutdown host identity mismatch")
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=30)
    require(gpu.returncode == 0 and not gpu.stdout.strip(), "GPU idle cannot be established")
    supervisors = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            comm = (proc / "comm").read_text().strip()
            command = (proc / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if comm == "supervisord" and b"/init/supervisor/supervisor.ini" in command:
            supervisors.append(int(proc.name))
    require(len(supervisors) == 1, "Expected exactly one native AutoDL supervisor")
    native = Path("/usr/bin/shutdown")
    body = native.read_bytes()
    require(b"supervisord" in body and b"xargs kill" in body, "Native provider changed; inspect before execution")
    receipt = {"time_utc": now(), "hostname": expected_hostname, "execute": execute,
               "supervisor_pid": supervisors[0], "native_shutdown_sha256": digest(native),
               "platform_off_independently_verified": False,
               "note": "Native stop request; no Trash deletion, no claim about billing"}
    write_json(Path(run_dir) / "shutdown-receipt.json", receipt)
    print(__import__("json").dumps({"native_shutdown_receipt": receipt}), flush=True)
    if execute:
        os.kill(supervisors[0], signal.SIGTERM)
    return receipt


def resolve_server(table_path, server_id):
    table = read_json(table_path)
    server = next((s for s in table["servers"] if s["id"] == server_id), None)
    require(server is not None, "Unknown server id")
    result = dict(server)
    for field in ("host", "identity_file", "known_hosts_file"):
        env_name = server.get(field + "_env")
        require(env_name and os.environ.get(env_name), "Set server environment variable: " + str(env_name))
        result[field] = os.environ[env_name]
    return result


def ssh_argv(server):
    # OpenSSH receives argv, never a constructed shell command containing credentials.
    return ["ssh", "-p", str(server["port"]), "-i", server["identity_file"],
            "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=yes",
            "-o", "UserKnownHostsFile=" + server["known_hosts_file"], server["user"] + "@" + server["host"]]
