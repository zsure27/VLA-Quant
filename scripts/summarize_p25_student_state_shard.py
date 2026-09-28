"""Pair a new closed-loop candidate with the frozen C0/C1/C2 episodes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from diagnostics.baseline_shard_analysis import parse_episodes


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--offset", required=True, type=int)
    p.add_argument("--count", required=True, type=int)
    args = p.parse_args()
    expected = {(task, init) for task in range(10) for init in range(args.offset, args.offset + args.count)}
    original = [json.loads(line) for line in (args.reference / "paired-episodes.jsonl").read_text().splitlines()]
    controls = {(row["task_id"], row["init_state_index"]): row for row in original if
                (row["task_id"], row["init_state_index"]) in expected}
    assert set(controls) == expected
    logs = list((args.candidate / "C3-12L-student-state").glob("EVAL-*.txt"))
    if len(logs) != 1 or (args.candidate / "C3-12L-student-state/exit-code.txt").read_text().strip() != "0":
        raise RuntimeError("Candidate evaluator incomplete")
    observed = parse_episodes(logs[0].read_text())
    candidate = {(row["task_id"], row["init_state_index"]): row for row in observed}
    if set(candidate) != expected:
        raise RuntimeError("Candidate initial-state coverage mismatch")
    rows = []
    for identity in sorted(expected):
        old, new = controls[identity], candidate[identity]
        for field in ("task_id", "init_state_index", "model_seed", "env_seed", "init_state_sha256"):
            if old[field] != new[field]:
                raise RuntimeError(f"Paired manifest mismatch {identity} {field}")
        if new["episode_errors"]:
            raise RuntimeError(f"Candidate evaluator error {identity}: {new['episode_errors']}")
        rows.append({**old, "C3": new["success"], "C3_episode_errors": new["episode_errors"]})
    summary = {"schema_version": "1.0", "classification": "development paired with frozen evaluator and exact manifest",
               "offset": args.offset, "count": args.count, "episodes": len(rows),
               "successes": {case: sum(row[case] for row in rows) for case in ("C0", "C1", "C2", "C3")},
               "frozen_reference": str(args.reference), "reference_pairs_sha256": digest(args.reference / "paired-episodes.jsonl"),
               "candidate_eval_sha256": digest(logs[0]),
               "leakage_contract": "only reset5-49; reset0-4 excluded due to student-state training",
               "holdout_touched": False}
    (args.candidate / "paired-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.candidate / "paired-episodes.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
