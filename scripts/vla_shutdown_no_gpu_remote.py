"""Request native AutoDL shutdown for a verified, GPU-less preparation instance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backup-dir", type=Path, required=True)
    p.add_argument("--expected-hostname", required=True)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if socket.gethostname() != a.expected_hostname:
        raise ValueError("Instance hostname mismatch")
    base = Path("/root/autodl-tmp/qvla-repro/backups").resolve(strict=True)
    backup = a.backup_dir.resolve(strict=True)
    if base not in backup.parents:
        raise ValueError("Backup must be a concrete persistent directory")
    counts = {}
    for name, expected in (("SHA256SUMS.txt", 13), ("RESULTS_SHA256SUMS.txt", 18494)):
        lines = (backup / name).read_text(encoding="utf-8").splitlines()
        if len(lines) != expected:
            raise ValueError("Incomplete archive manifest")
        for line in lines:
            expected_sha, relative = line.split(maxsplit=1)
            path = (backup / relative.lstrip("*")).resolve(strict=True)
            if backup not in path.parents or digest(path) != expected_sha:
                raise ValueError("Archive SHA/path mismatch")
        counts[name] = len(lines)
    scope = json.loads((backup / "SERVER_ARCHIVE_SCOPE.json").read_text(encoding="utf-8"))
    if not (backup / f"VLA-Quant-{scope['server_commit']}.bundle").is_file():
        raise ValueError("Original Git bundle absent")
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                         text=True, capture_output=True, check=False)
    listing = subprocess.run(["nvidia-smi", "-L"], text=True, capture_output=True, check=False)
    if not (gpu.returncode != 0 and gpu.stdout.strip().lower() == "no devices were found"
            and listing.returncode == 0 and listing.stdout.strip().lower() == "no devices found."):
        raise ValueError("Current instance is not confirmed GPU-less and idle")
    supervisors = []
    active_experiments = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            name = (proc / "comm").read_text().strip()
            command = (proc / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if name == "supervisord" and b"/init/supervisor/supervisor.ini" in command:
            supervisors.append(int(proc.name))
        if any(any(marker in part for marker in (b"vla_stage_supervisor.py", b"train_b4_joint_peft.py",
                                             b"run_eval_b4_joint.py", b"run_eval_official_quant.py")) for part in command):
            active_experiments.append(int(proc.name))
    if len(supervisors) != 1 or active_experiments:
        raise ValueError("Supervisor identity or experiment-idle check failed")
    native = Path("/usr/bin/shutdown").read_bytes()
    if b"supervisord" not in native or b"xargs kill" not in native:
        raise ValueError("Native shutdown implementation changed")
    receipt = {"time_utc": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
               "backup": str(backup), "verified_counts": counts,
               "supervisor_pid": supervisors[0], "native_shutdown_sha256": hashlib.sha256(native).hexdigest(),
               "gpu_devices_found": False, "active_experiments": [], "execute": a.execute,
               "platform_off_independently_verified": False,
               "note": "Native supervisor stop request; omits Trash deletion"}
    print(json.dumps(receipt), flush=True)
    if a.execute:
        (backup / "B4_PRERUN_SHUTDOWN_REQUEST.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        os.kill(supervisors[0], signal.SIGTERM)


if __name__ == "__main__":
    main()
