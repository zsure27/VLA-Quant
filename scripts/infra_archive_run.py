"""Snapshot this round only; stream SHA-verified chunks to the approved private archive.

No deletion. Transfer budget is bounded before uploading. Mutable closure logs are
excluded from the snapshot and retained on persistent disk for the final receipt.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tarfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qvla.infra.common import digest, read_json, write_json, now


def collect(root):
    files = []
    banned = re.compile(r"(^\.env($|\.)|credential|known_hosts|id_ed25519|id_rsa|token|secret)", re.I)
    marker = re.compile(rb"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")
    for item in sorted(root.rglob("*")):
        relative = item.relative_to(root)
        if ".closure" in relative.parts or item.name in ("backup.log", "git-sync.log"):
            continue
        if item.is_symlink():
            raise ValueError("Archive refuses symlinks")
        if not item.is_file():
            continue
        if any(banned.search(x) for x in relative.parts):
            raise ValueError("Credential-like filename in raw run; audit it before backup")
        if item.suffix in (".json", ".jsonl", ".log", ".txt", ".py", ".md"):
            with item.open("rb") as stream:
                carry = b""
                for block in iter(lambda: stream.read(1024**2), b""):
                    if marker.search(carry + block):
                        raise ValueError("Credential marker in run material")
                    carry = block[-512:]
        files.append({"path": relative.as_posix(), "bytes": item.stat().st_size, "sha256": digest(item)})
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--max-upload-gib", type=float, default=2)
    p.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    p.add_argument("--prepare-only", action="store_true", help="Build/verify snapshot and print transfer budget without network")
    a = p.parse_args()
    if not math.isfinite(a.max_upload_gib) or a.max_upload_gib <= 0:
        raise ValueError("Upload budget must be a finite positive GiB value")
    root = a.run_dir.resolve(strict=True)
    if root == Path(root.anchor) or not (root / "status.json").is_file():
        raise ValueError("Require one concrete completed/interrupted round directory")
    if read_json(root / "status.json")["state"] == "RUNNING":
        raise ValueError("Stop the round before snapshotting raw files")
    files = collect(root)
    if any(row["path"] == "ARCHIVE_MEMBERS.json" for row in files):
        raise ValueError("ARCHIVE_MEMBERS.json is a reserved archive member name")
    if shutil.disk_usage(root).free < sum(row["bytes"] for row in files) + 512 * 1024**2:
        raise ValueError("Insufficient disk for a conservative snapshot; keep original persistent data")
    area = root / ".closure"
    area.mkdir(exist_ok=True)
    write_json(area / "members.json", {"members": files, "time_utc": now()})
    archive = area / "round.tar.gz"
    if archive.exists():
        raise ValueError("Snapshot already exists; reuse its saved upload plan instead of recompressing")
    with tarfile.open(archive, "w:gz") as tar:
        for record in files:
            item = root / record["path"]
            if digest(item) != record["sha256"]:
                raise ValueError("Source changed during archive")
            tar.add(item, arcname=record["path"], recursive=False)
        tar.add(area / "members.json", arcname="ARCHIVE_MEMBERS.json")
    with tarfile.open(archive, "r:gz") as tar:
        import hashlib
        expected = {r["path"]: r for r in files}
        for member in tar:
            if member.name == "ARCHIVE_MEMBERS.json":
                continue
            h = hashlib.sha256()
            stream = tar.extractfile(member)
            for block in iter(lambda: stream.read(4 * 1024**2), b""):
                h.update(block)
            if member.size != expected[member.name]["bytes"] or h.hexdigest() != expected[member.name]["sha256"]:
                raise ValueError("Archive member roundtrip mismatch")
    size = archive.stat().st_size
    if size > a.max_upload_gib * 1024**3:
        raise ValueError(f"Upload budget exceeded: {size} bytes; minimum upload+readback is {2*size} bytes. Snapshot kept.")
    plan = {"repository": "zsure27/VLA-Quant-Archives", "security_scan_passed": True,
            "files": [{"path": str(archive), "bytes": size, "sha256": digest(archive)}]}
    write_json(area / "upload-plan.json", plan)
    if a.prepare_only:
        print(json.dumps({"prepared": True, "archive_bytes": size, "members": len(files),
                          "minimum_transfer_bytes": size * 2, "network_used": False}))
        return
    spec = importlib.util.spec_from_file_location("_infra_archive_transport", ROOT / "scripts/github_archive_store_v2.py")
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    parts = core.prepare_parts(plan, chunk_size=256 * 1024**2)
    client = core.GitHub(core.credential_token())  # Verifies private destination; secrets stay in memory.
    receipt = core.Receipt(area / "remote-verification.json", client.release)
    core.run_parts(parts, client, receipt, workers=a.workers)
    write_json(root / "backup-receipt.json", {"time_utc": now(), "archive_sha256": plan["files"][0]["sha256"],
               "archive_bytes": size, "members": len(files), "remote_download_sha256_verified": True,
               "release": client.release["html_url"], "parts": [p.name for p in parts],
               "local_data_deleted": False})


if __name__ == "__main__":
    main()
