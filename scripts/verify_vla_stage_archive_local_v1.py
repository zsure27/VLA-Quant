"""Restore a trusted stage tar safely and verify every recorded local file."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(part)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--expected-bytes", type=int, required=True)
    p.add_argument("--expected-entries", type=int, required=True)
    a = p.parse_args()
    display_root = a.directory.resolve()
    root = Path("\\\\?\\" + str(display_root)) if os.name == "nt" else display_root
    archive = root / "complete.tar.gz"
    if archive.stat().st_size != a.expected_bytes or digest(archive) != a.expected_sha256:
        raise ValueError("Downloaded tar size/SHA mismatch; retain partial transfer")
    server = json.loads((root / "SERVER_VERIFICATION.json").read_text())
    if (server["sha256"] != a.expected_sha256 or server["bytes"] != a.expected_bytes
            or server["sha_entries"] != a.expected_entries or not server["all_tar_members_verified"]):
        raise ValueError("Server receipt does not match the trusted transfer anchor")
    with tarfile.open(archive, "r:gz") as tar:
        stream = tar.extractfile("SHA256SUMS.txt")
        if stream is None:
            raise ValueError("Archive checksum list missing")
        checksum_bytes = stream.read()
        sums = dict((name, digest_value) for digest_value, name in
                    (row.split("  ", 1) for row in checksum_bytes.decode().splitlines()))
        if len(sums) != a.expected_entries:
            raise ValueError("Archive checksum count mismatch")
        expected = dict(sums)
        expected["SHA256SUMS.txt"] = hashlib.sha256(checksum_bytes).hexdigest()
        seen = set()
        for member in tar:
            relative = PurePosixPath(member.name)
            if (not member.isfile() or relative.is_absolute() or ".." in relative.parts
                    or "\\" in member.name or ":" in member.name or member.name not in expected
                    or member.name in seen):
                raise ValueError(f"Unexpected or unsafe tar member: {member.name}")
            destination = root.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                with tar.extractfile(member) as source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output, length=4 * 1024**2)
            if digest(destination) != expected[member.name]:
                raise ValueError(f"Local restored file SHA mismatch: {member.name}")
            seen.add(member.name)
        if seen != set(expected):
            raise ValueError("Restored local archive incomplete")
    receipt = {"gate": "PASS_COMPLETED_STAGE_LOCAL_ARCHIVE_ALL_FILES", "time_utc": datetime.now(timezone.utc).isoformat(),
               "directory": str(display_root), "bytes": a.expected_bytes, "sha256": a.expected_sha256,
               "sha_entries": a.expected_entries, "restored_members": len(seen),
               "all_files_verified": True, "active_evaluations_excluded": True}
    (root / "LOCAL_VERIFICATION.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
