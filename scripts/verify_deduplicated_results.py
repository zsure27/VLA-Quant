"""Verify the exact-byte replacement map for removed duplicate result files."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "results/DEDUPLICATED_FILES_20260922.json"


def inside_root(relative_path):
    path = (ROOT / relative_path).resolve()
    if ROOT not in path.parents or path == ROOT:
        raise ValueError("path outside repository: " + relative_path)
    return path


def verify():
    manifest = json.loads(MAP.read_text(encoding="utf-8"))
    canonical = {entry["path"]: entry for entry in manifest["canonical_files"]}
    if len(canonical) != len(manifest["canonical_files"]):
        raise ValueError("duplicate canonical path")
    for name, entry in canonical.items():
        path = inside_root(name)
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError("missing or changed canonical file: " + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("canonical SHA256 mismatch: " + name)
    old_paths = set()
    for entry in manifest["removed_exact_duplicates"]:
        old = entry["old_path"]
        if old in old_paths or inside_root(old).exists():
            raise ValueError("old path duplicated or still present: " + old)
        old_paths.add(old)
        target = canonical.get(entry["replacement"])
        if not target or (entry["sha256"], entry["bytes"]) != (target["sha256"], target["bytes"]):
            raise ValueError("invalid replacement mapping: " + old)
    return {"canonical_files": len(canonical), "removed_exact_duplicates": len(old_paths),
            "removed_bytes": sum(x["bytes"] for x in manifest["removed_exact_duplicates"]),
            "retained_bytes": sum(x["bytes"] for x in canonical.values())}


if __name__ == "__main__":
    print(json.dumps(verify()))
