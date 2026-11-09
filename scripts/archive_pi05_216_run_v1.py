"""Archive the 216 trajectory study's unique evidence, excluding public model weights.

The archive is immutable once created. It includes all run outputs, stage logs,
checks and the scripts sent directly to the server during the round. The
official GCS checkpoint is recorded by its verified download receipt and can
be retrieved by source object identity; this script never claims it is inside
the private archive.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def digest(stream) -> str:
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve(strict=True)
    archive = args.archive.resolve()
    if archive.exists() or archive.is_relative_to(root):
        raise ValueError("Archive must be a new file outside the source root")
    globs = (
        "runs/20261108-216-pi05-*/**/*",
        "logs/*pi05*", "logs/*216*",
        "checks/*pi05*", "checks/*216*", "checks/*openpi*", "checks/*pi05*/**/*",
        "models/pi05_libero/DOWNLOAD_VERIFICATION.json",
        "closure/20261109-216-bridge-blocked-v1/**/*",
        "archive_pi05_216_run_v1.py",
        "launch_pi05_crosssuite_bf16_v1.sh",
        "convert_pi05_pytorch_bridge_v1.sh",
        "pi05_same_input_bridge_v*.py", "audit_pi05_conversion_mapping_*.py", "launch_pi05_torch_bridge_*.sh", "convert_pi05_pytorch_bridge_v*.sh",
        "20261108-216-pi05-torch-bridge-v1.json",
    )
    files: dict[str, Path] = {}
    for pattern in globs:
        for path in root.glob(pattern):
            if path.is_dir():
                continue
            resolved = path.resolve(strict=True)
            if path.is_symlink() or not resolved.is_relative_to(root) or not resolved.is_file():
                raise ValueError(f"Unsafe archive member: {path}")
            relative = path.relative_to(root).as_posix()
            files[relative] = path
    required = {
        "models/pi05_libero/DOWNLOAD_VERIFICATION.json",
        "launch_pi05_crosssuite_bf16_v1.sh",
        "pi05_same_input_bridge_v1.py",
        "20261108-216-pi05-torch-bridge-v1.json",
    }
    if not required.issubset(files) or len([x for x in files if x.endswith("/result.json") and x.startswith("runs/")]) < 4:
        raise ValueError("Missing required source identities or native baseline results")
    rows = []
    for relative, path in sorted(files.items()):
        with path.open("rb") as stream:
            value = digest(stream)
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": value})
    manifest = {
        "archive_kind": "216-pi05-trajectory-evidence-v1",
        "members": len(rows),
        "files": rows,
        "excluded_retrievable_parent": "models/pi05_libero checkpoint bytes (12,439,085,481); official GCS source and per-object hashes in DOWNLOAD_VERIFICATION.json",
        "excluded_reproducible_derivative": "models/pi05_libero_pytorch_bridge_v1; regenerate with included source script only if its numerical gate passed",
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        for relative, path in sorted(files.items()):
            tar.add(path, arcname=relative, recursive=False)
        info = tarfile.TarInfo("MANIFEST.json")
        info.size = len(manifest_bytes)
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(manifest_bytes))
    verified = 0
    with tarfile.open(archive, "r:gz") as tar:
        for row in rows:
            member = tar.getmember(row["path"])
            if member.size != row["bytes"]:
                raise ValueError(f"Member length differs: {row['path']}")
            with tar.extractfile(member) as stream:
                if digest(stream) != row["sha256"]:
                    raise ValueError(f"Member SHA differs: {row['path']}")
            verified += 1
        if tar.extractfile("MANIFEST.json").read() != manifest_bytes:
            raise ValueError("Manifest differs")
    with archive.open("rb") as stream:
        archive_sha = digest(stream)
    receipt = {"gate": "PASS_PI05_216_ARCHIVE_MEMBER_VERIFICATION", "archive": str(archive),
               "bytes": archive.stat().st_size, "sha256": archive_sha,
               "members": len(rows) + 1, "members_verified": verified + 1}
    archive.with_suffix(archive.suffix + ".receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
