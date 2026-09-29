"""Check the current-tree hashes in standardized VLA session manifests."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".md", ".py", ".json", ".jsonl", ".csv", ".txt", ".log", ".svg"}


def verify(path):
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    entries = document["files"]
    pairs = (entries.items() if isinstance(entries, dict)
             else ((row["path"], row["sha256"]) for row in entries))
    count = 0
    for relative, expected in pairs:
        source = (ROOT / relative).resolve()
        if ROOT not in source.parents or not source.is_file():
            raise ValueError("missing or external source: " + relative)
        content = source.read_bytes()
        if source.suffix in TEXT_SUFFIXES:
            content = content.replace(b"\r\n", b"\n")
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("SHA256 mismatch: " + relative)
        count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", help="Verify only this session directory name")
    args = parser.parse_args()
    counts = {}
    failures = {}
    for path in sorted((ROOT / "reports/experiments").glob("*/*/manifest.json")):
        if args.session and path.parent.name != args.session:
            continue
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        if document.get("files") is not None:
            try:
                counts[path.parent.name] = verify(path)
            except ValueError as error:
                failures[path.parent.name] = str(error)
    print(json.dumps({"sessions_verified": len(counts), "files_verified": sum(counts.values()),
                      "per_session": counts, "failures": failures}))
    if failures:
        sys.exit(1)
