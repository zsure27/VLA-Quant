"""Verify a downloaded VLA archive using its two SHA256 manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    archive = args.archive.resolve(strict=True)
    counts = {}
    for manifest_name in ("SHA256SUMS.txt", "RESULTS_SHA256SUMS.txt"):
        manifest = archive / manifest_name
        count = 0
        for line in manifest.read_text(encoding="utf-8").splitlines():
            expected, relative = line.split(maxsplit=1)
            path = (archive / relative.lstrip("*")).resolve(strict=True)
            if archive not in path.parents or digest(path) != expected:
                raise ValueError(f"Archive hash/path mismatch: {relative}")
            count += 1
        counts[manifest_name] = count
    scope = json.loads((archive / "SERVER_ARCHIVE_SCOPE.json").read_text(encoding="utf-8"))
    bundle = archive / f"VLA-Quant-{scope['server_commit']}.bundle"
    if not bundle.is_file():
        raise ValueError("Server Git bundle missing")
    print(json.dumps({"gate": "PASS_LOCAL_ARCHIVE_SHA256", "archive": str(archive),
                      "files": counts, "server_commit": scope["server_commit"]}))


if __name__ == "__main__":
    main()
