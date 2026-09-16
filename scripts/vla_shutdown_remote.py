"""Use the observed AutoDL container shutdown mechanism without clearing Trash.

This sends the platform-native supervisor stop request. Control-panel OFF/billing
status must still be independently confirmed; losing SSH is not that proof.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup-dir", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    base = Path("/root/autodl-tmp/qvla-repro/backups").resolve()
    backup = args.backup_dir.resolve()
    if backup == base or base not in backup.parents:
        raise SystemExit("Backup must be a concrete directory under the persistent backup root")
    for manifest_name in ("SHA256SUMS.txt", "RESULTS_SHA256SUMS.txt"):
        for line in (backup / manifest_name).read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            path = (backup / name.lstrip("*")).resolve()
            if backup not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise SystemExit("Backup hash/path verification failed")
    commit = subprocess.check_output(["git", "-C", "/root/VLA-Quant", "rev-parse", "--short", "HEAD"], text=True).strip()
    if not (backup / f"VLA-Quant-{commit}.bundle").is_file():
        raise SystemExit("Backup does not contain current commit")
    gpu_processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if gpu_processes:
        raise SystemExit("GPU processes still active; save and stop them before shutdown")
    supervisors = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            name = (process / "comm").read_text().strip()
            cmdline = (process / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if name == "supervisord" and b"/init/supervisor/supervisor.ini" in cmdline:
            supervisors.append(int(process.name))
    if len(supervisors) != 1:
        raise SystemExit("Expected exactly one known AutoDL supervisor")
    native = Path("/usr/bin/shutdown").read_bytes()
    if b"supervisord" not in native or b"xargs kill" not in native:
        raise SystemExit("Native shutdown implementation changed; inspect it before use")
    receipt = {"time_utc": datetime.now(timezone.utc).isoformat(), "commit": commit,
               "backup": str(backup), "supervisor_pid": supervisors[0],
               "native_shutdown_sha256": hashlib.sha256(native).hexdigest(),
               "execute": args.execute, "control_panel_off_verified": False,
               "note": "Uses native supervisor termination; omits native Trash deletion"}
    print(json.dumps(receipt), flush=True)
    if args.execute:
        (backup / "SHUTDOWN_REQUEST.json").write_text(json.dumps(receipt, indent=2) + "\n")
        with open("/proc/1/fd/1", "w") as stream:
            stream.write("[" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "] user execute shutdown command in container!\n")
            stream.flush()
        os.kill(supervisors[0], signal.SIGTERM)


if __name__ == "__main__":
    main()
