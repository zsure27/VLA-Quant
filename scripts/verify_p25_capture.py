"""Verify a completed student observation capture before a teacher-only retry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("student_run", type=Path)
    args = parser.parse_args()
    if (args.student_run.parent / "student-rollout-exit-code.txt").read_text().strip() != "0":
        raise RuntimeError("Student rollout did not exit successfully")
    rows = [json.loads(line) for line in (args.student_run / "on-policy-events.jsonl").read_text().splitlines()]
    queries = [row for row in rows if row["record_type"] == "query"]
    outcomes = [row for row in rows if row["record_type"] == "episode_end"]
    if len(outcomes) != 10 or len(queries) < 10 or any(row["aborted"] for row in outcomes):
        raise RuntimeError("Capture is incomplete or aborted")
    if set(row["episode_serial"] for row in queries) != set(range(10)):
        raise RuntimeError("Query episode identities are incomplete")
    if set(row["episode_serial"] for row in outcomes) != set(range(10)):
        raise RuntimeError("Outcome episode identities are incomplete")
    for row in queries:
        path = args.student_run / row["file"]
        if sha256(path) != row["file_sha256"]:
            raise RuntimeError(f"Sample hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as sample:
            action = sample["student_action"]
            if action.shape != (8, 7) or not np.isfinite(action).all():
                raise RuntimeError(f"Invalid action chunk: {path}")
    print(json.dumps({
        "episodes": len(outcomes), "queries": len(queries),
        "successes": sum(bool(row["success"]) for row in outcomes),
        "events_sha256": sha256(args.student_run / "on-policy-events.jsonl"),
        "all_sample_sha256_verified": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
