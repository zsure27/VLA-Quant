"""Make a checksum-verified persistent closure directory without duplicating disk blocks.

The source artifacts remain untouched. The closure directory hard-links regular files
on the same persistent filesystem; a separate local copy is required for redundancy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
from datetime import datetime, timezone

BASE = Path("/root/autodl-tmp/qvla-repro/backups").resolve()
REPO = Path("/root/VLA-Quant")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def within(path: Path, parent: Path) -> bool:
    return path != parent and parent in path.parents


def link_files(source: Path, destination: Path) -> list[Path]:
    if not within(source, BASE) or not source.is_dir() or source.is_symlink():
        raise ValueError(f"Invalid source: {source}")
    linked: list[Path] = []
    for root, dirs, files in os.walk(source, followlinks=False):
        root_path = Path(root)
        if any((root_path / dirname).is_symlink() for dirname in dirs):
            raise ValueError(f"Source contains a symlink directory: {root_path}")
        relative = root_path.relative_to(source)
        target_dir = destination / relative
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            path = root_path / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Source contains a nonregular file: {path}")
            if path.parent == source and name in ("SHA256SUMS.txt", "RESULTS_SHA256SUMS.txt", "SHUTDOWN_REQUEST.json"):
                continue
            target = target_dir / name
            os.link(path, target)
            linked.append(target)
    return linked


def manifest(output: Path, files: list[Path], name: str) -> None:
    with (output / name).open("x", encoding="utf-8", newline="\n") as stream:
        for path in sorted(files):
            stream.write(f"{sha256(path)}  {path.relative_to(output).as_posix()}\n")


def verify(output: Path, name: str) -> int:
    count = 0
    for line in (output / name).read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = (output / relative.lstrip("*")).resolve()
        if not within(path, output) or sha256(path) != expected:
            raise ValueError(f"Hash verification failed: {relative}")
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-hostname", required=True)
    parser.add_argument("--existing", required=True, type=Path)
    parser.add_argument("--session25", required=True, type=Path)
    parser.add_argument("--session30", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if socket.gethostname() != args.expected_hostname:
        raise ValueError("046 host identity mismatch")
    output = args.output.resolve()
    if not within(output, BASE) or output.exists():
        raise ValueError("Output must be a new concrete directory under persistent backups")
    import subprocess
    commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()
    existing = args.existing.resolve()
    if not (existing / f"VLA-Quant-{commit}.bundle").is_file():
        raise ValueError("Existing archive lacks the current server repository bundle")
    output.mkdir(parents=True)
    old_files = link_files(existing, output)
    new_files = link_files(args.session25.resolve(), output / "boundary-25-29-v1")
    new_files += link_files(args.session30.resolve(), output / "boundary-30-39-v1")
    scope = output / "SERVER_ARCHIVE_SCOPE.json"
    scope.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": args.expected_hostname,
        "server_commit": commit,
        "source_archive": str(existing),
        "source_sessions": [str(args.session25), str(args.session30)],
        "storage": "same-filesystem hard links, no deletion or rewrite of source files",
        "redundancy": "copy the entire closure directory to local storage and verify both manifests",
        "excluded": "deterministic B3 SVD spool, retained at its original persistent path",
    }, indent=2) + "\n", encoding="utf-8")
    old_files.append(scope)
    manifest(output, old_files, "SHA256SUMS.txt")
    manifest(output, new_files, "RESULTS_SHA256SUMS.txt")
    counts = {name: verify(output, name) for name in ("SHA256SUMS.txt", "RESULTS_SHA256SUMS.txt")}
    print(json.dumps({"gate": "PASS_PERSISTENT_ARCHIVE", "output": str(output), "files": counts,
                      "bundle": f"VLA-Quant-{commit}.bundle"}))


if __name__ == "__main__":
    main()
