"""Verify the archived Response-SVD inputs against the trajectory split and bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(directory: Path, split_path: Path) -> dict:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    episodes = {row["dataset_order_index"]: row for row in split["episodes"]}
    rows = manifest["samples"]
    assert len(rows) == 80
    assert len({row["episode"] for row in rows}) == 80
    assert len({row["file"] for row in rows}) == 80
    assert {row["file"] for row in rows} == {
        path.name for path in directory.glob("sample-*.npz")
    }
    roles = Counter()
    for row in rows:
        trajectory = episodes[row["episode"]]
        assert trajectory["instruction"] == row["instruction"]
        assert trajectory["steps"] == row["steps"]
        assert row["step"] < row["steps"]
        assert row["trajectory_role"] == trajectory["role"]
        assert sha256(directory / row["file"]) == row["sha256"]
        roles[trajectory["role"]] += 1
    assert roles == {"peft_train": 80}, roles
    return {
        "gate": "PASS_SVD_CALIBRATION80_SOURCE",
        "manifest_sha256": sha256(manifest_path),
        "trajectory_split_sha256": sha256(split_path),
        "sample_count": len(rows),
        "unique_trajectories": len({row["episode"] for row in rows}),
        "trajectory_roles": dict(roles),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory, args.split), sort_keys=True))
