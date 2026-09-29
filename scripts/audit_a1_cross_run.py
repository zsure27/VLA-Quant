"""Check whether an A1 seed variant produced independent episode outcomes."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path


CASES = ("C0", "C1", "C2", "C3")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_archive", type=Path)
    parser.add_argument("new_results", type=Path)
    args = parser.parse_args()
    reference = {}
    with tarfile.open(args.reference_archive, "r:gz") as archive:
        for member in archive.getmembers():
            if member.name.startswith("eval/p25-student-state80-") and member.name.endswith("/paired-episodes.jsonl"):
                for line in archive.extractfile(member):
                    row = json.loads(line)
                    key = (row["task_id"], row["init_state_index"])
                    if key in reference:
                        raise ValueError("duplicate reference episode: %r" % (key,))
                    reference[key] = row
    candidate = {}
    for path in args.new_results.glob("paired-*-episodes.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (row["task_id"], row["init_state_index"])
            if key in candidate:
                raise ValueError("duplicate candidate episode: %r" % (key,))
            candidate[key] = row
    expected = {(task, reset) for task in range(10) for reset in range(5, 50)}
    if set(reference) != expected or set(candidate) != expected:
        raise ValueError("reference or candidate does not cover exactly 450 task/reset pairs")
    result = {
        "episodes_compared": len(expected),
        "model_seed_changed": sum(reference[k]["model_seed"] != candidate[k]["model_seed"] for k in expected),
        "env_seed_changed": sum(reference[k]["env_seed"] != candidate[k]["env_seed"] for k in expected),
        "initial_state_sha_changed": sum(reference[k]["init_state_sha256"] != candidate[k]["init_state_sha256"] for k in expected),
        "success_outcome_changed": {case: sum(bool(reference[k][case]) != bool(candidate[k][case])
                                              for k in expected) for case in CASES},
        "classification": "cross-run outcome identity audit; outcome identity alone does not prove identical observations",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
