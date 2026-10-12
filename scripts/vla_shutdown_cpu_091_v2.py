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
import socket
import subprocess
from datetime import datetime, timezone


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup-dir", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-hostname", required=True)
    args = parser.parse_args()
    if socket.gethostname() != args.expected_hostname:
        raise SystemExit("Actual hostname differs from the user-confirmed CPU instance")
    base = Path("/root/autodl-tmp/qvla-repro/backups").resolve()
    backup = args.backup_dir.resolve()
    if backup == base or base not in backup.parents:
        raise SystemExit("Backup must be a concrete directory under the persistent backup root")
    counts = {}
    for manifest_name in ("SHA256SUMS.txt", "RESULTS_SHA256SUMS.txt"):
        lines = (backup / manifest_name).read_text().splitlines()
        if not lines:
            raise SystemExit("Empty archive manifest")
        for line in lines:
            expected_digest, name = line.split(maxsplit=1)
            path = (backup / name.lstrip("*")).resolve()
            if backup not in path.parents or digest(path) != expected_digest:
                raise SystemExit("Backup hash/path verification failed")
        counts[manifest_name] = len(lines)
    commit = subprocess.check_output(["git", "-C", "/root/VLA-Quant", "rev-parse", "--short", "HEAD"], text=True).strip()
    if not (backup / f"VLA-Quant-{commit}.bundle").is_file():
        raise SystemExit("Backup does not contain current commit")
    # No-GPU AutoDL boots install a zero-byte nvidia-smi placeholder. An empty
    # shell command's exit 0 is not proof of GPU idleness. This separate helper
    # accepts only the inspected placeholder AND actual zero assigned devices.
    gpu_stub = Path("/usr/bin/nvidia-smi")
    if gpu_stub.resolve() != gpu_stub or gpu_stub.read_bytes() != b"":
        raise SystemExit("No-GPU placeholder differs; use the standard GPU helper")
    devices = list(Path("/dev").glob("nvidia*"))
    import torch
    available, count = torch.cuda.is_available(), torch.cuda.device_count()
    if devices or available or count != 0:
        raise SystemExit("An assigned GPU/device exists; use the standard GPU helper")
    gpu_receipt = {"mode": "confirmed_no_assigned_gpu", "cuda_available": available,
                   "cuda_device_count": count, "nvidia_device_nodes": [],
                   "nvidia_smi_bytes": 0,
                   "nvidia_smi_sha256": hashlib.sha256(b"").hexdigest()}
    supervisors = []
    active_experiments = []
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
        if any(any(marker in part for marker in (b"vla_stage_supervisor.py", b"run_eval", b"train_b5", b"label_b5", b"probe_qvla", b"train_b4")) for part in cmdline):
            active_experiments.append(int(process.name))
    if len(supervisors) != 1 or active_experiments:
        raise SystemExit("Supervisor identity or experiment idle check failed")
    native = Path("/usr/bin/shutdown").read_bytes()
    if b"supervisord" not in native or b"xargs kill" not in native:
        raise SystemExit("Native shutdown implementation changed; inspect it before use")
    receipt = {"time_utc": datetime.now(timezone.utc).isoformat(), "commit": commit,
               "hostname": socket.gethostname(), "gpu_admission": gpu_receipt,
               "verified_counts": counts, "active_experiments": [],
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
