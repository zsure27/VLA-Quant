"""Check the current-tree hashes in standardized VLA session manifests."""

import hashlib
import json
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
    counts = {}
    for path in sorted((ROOT / "reports/experiments").glob("*/*/manifest.json")):
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        if document.get("files") is not None:
            counts[path.parent.name] = verify(path)
    print(json.dumps({"sessions_verified": len(counts), "files_verified": sum(counts.values()),
                      "per_session": counts}))
