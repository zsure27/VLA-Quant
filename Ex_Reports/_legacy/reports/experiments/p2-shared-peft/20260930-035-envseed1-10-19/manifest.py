#!/usr/bin/env python3
"""Create or verify the paired report/results SHA256 manifest for this session."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

REPORT = Path(__file__).resolve().parent
REPO = REPORT.parents[3]
RESULTS = REPO / "results/experiments/p2-shared-peft" / REPORT.name
MANIFEST = REPORT / "manifest.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect() -> list[dict[str, object]]:
    files = []
    for base in (REPORT, RESULTS):
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            if path == MANIFEST:
                continue
            files.append({
                "path": path.relative_to(REPO).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            })
    return sorted(files, key=lambda item: str(item["path"]))


def main() -> int:
    expected = {"schema_version": 1, "session": REPORT.name, "files": collect()}
    if "--verify" in sys.argv[1:]:
        actual = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if actual != expected:
            raise SystemExit("manifest mismatch")
        print(f"MANIFEST_VERIFIED files={len(expected['files'])}")
    else:
        MANIFEST.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
        print(f"MANIFEST_WRITTEN files={len(expected['files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
