"""Verify both closure manifests and the expected 450 original rollout videos."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    base = args.directory.resolve(strict=True)
    checked = []
    for manifest in ("SHA256SUMS.txt", "RESULTS_SHA256SUMS.txt"):
        for line in (base / manifest).read_text().splitlines():
            expected, name = line.split(maxsplit=1)
            path = (base / name.lstrip("*")).resolve(strict=True)
            assert base in path.parents, path
            assert digest(path) == expected, path
            checked.append(name)
    videos = json.loads((base / "VIDEO_MANIFEST.json").read_text())
    assert len(videos) == 450, len(videos)
    assert len({v["archive_name"] for v in videos}) == 450
    with tarfile.open(base / "session-results.tar.gz") as archive:
        names = set(archive.getnames())
        assert all(v["archive_name"] in names for v in videos)
        assert sum("/paired-episodes.jsonl" in n for n in names) == 5
    print(json.dumps({"archive": str(base), "verified_files": len(checked),
                      "original_videos": len(videos), "raw_paired_shards": 5,
                      "session_results_sha256": digest(base / "session-results.tar.gz")}, indent=2))


if __name__ == "__main__":
    main()
