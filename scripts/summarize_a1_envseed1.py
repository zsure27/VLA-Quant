"""Summarize an A1 shard only from completed, manifest-paired evaluator outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from diagnostics.baseline_shard_analysis import parse_episodes

CASES = ("C0", "C3", "C2", "C1", "BF16")
IDENTITY = ("task_id", "init_state_index", "model_seed", "env_seed", "init_state_sha256")


def load_case(base: Path, case: str, expected: set[tuple[int, int]]):
    directory = base / case
    if not (directory / "complete.json").is_file():
        return None
    if (directory / "exit-code.txt").read_text().strip() != "0":
        raise ValueError(f"{case}: nonzero exit")
    logs = list(directory.glob("EVAL-*.txt"))
    if len(logs) != 1:
        raise ValueError(f"{case}: expected exactly one evaluator log")
    rows = parse_episodes(logs[0].read_text())
    mapping = {(row["task_id"], row["init_state_index"]): row for row in rows}
    if len(rows) != len(expected) or set(mapping) != expected:
        raise ValueError(f"{case}: incomplete or duplicate manifest")
    if any(row["episode_errors"] for row in rows):
        raise ValueError(f"{case}: evaluator episode errors")
    return mapping, hashlib.sha256(logs[0].read_bytes()).hexdigest()


def exact_mcnemar(rescue: int, new_failure: int) -> float:
    n = rescue + new_failure
    if n == 0:
        return 1.0
    low = min(rescue, new_failure)
    return min(1.0, 2.0 * sum(math.comb(n, k) for k in range(low + 1)) / 2**n)


def stratified_interval(rows: list[dict], trials: int = 20000) -> list[float]:
    rng = random.Random(20260929)
    groups = [[int(row["C3"]) - int(row["C0"]) for row in rows if row["task_id"] == task]
              for task in range(10)]
    samples = []
    for _ in range(trials):
        total = sum(rng.choice(group) for group in groups for _ in range(len(group)))
        samples.append(total / len(rows))
    samples.sort()
    return [samples[int(0.025 * trials)], samples[int(0.975 * trials)]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("--offset", type=int, default=5)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    expected = {(task, reset) for task in range(10)
                for reset in range(args.offset, args.offset + args.count)}
    loaded = {case: load_case(args.base, case, expected) for case in CASES}
    present = [case for case in CASES if loaded[case] is not None]
    if args.require_all and len(present) != len(CASES):
        raise ValueError(f"missing cases: {set(CASES) - set(present)}")
    if not {"C0", "C3"}.issubset(present):
        raise ValueError("C0 and C3 must both be complete before paired analysis")
    paired = []
    for key in sorted(expected):
        reference = loaded["C0"][0][key]
        row = {name: reference[name] for name in IDENTITY}
        for case in present:
            episode = loaded[case][0][key]
            if tuple(episode[name] for name in IDENTITY) != tuple(reference[name] for name in IDENTITY):
                raise ValueError(f"unpaired manifest: {case} {key}")
            row[case] = episode["success"]
        paired.append(row)
    rescue = sum(not row["C0"] and row["C3"] for row in paired)
    new_failure = sum(row["C0"] and not row["C3"] for row in paired)
    successes = {case: sum(row[case] for row in paired) for case in present}
    by_task = {str(task): {case: sum(row[case] for row in paired if row["task_id"] == task)
                           for case in present} for task in range(10)}
    headroom = successes["C2"] - successes["C0"] if "C2" in present else None
    summary = {
        "classification": "A1 same-official-initial-state different-env-randomness pilot; not new initial states or blind holdout",
        "offset": args.offset, "count_per_task": args.count, "episodes_per_case": len(paired),
        "configs_complete": present, "successes": successes, "by_task": by_task,
        "C3_minus_C0": successes["C3"] - successes["C0"], "rescue": rescue,
        "new_failure": new_failure, "exact_mcnemar_p": exact_mcnemar(rescue, new_failure),
        "paired_stratified_bootstrap_95ci_fraction": stratified_interval(paired),
        "C2_minus_C0_headroom": headroom,
        "W4_recovery_fraction": ((successes["C3"] - successes["C0"]) / headroom
                                 if headroom is not None and headroom > 0 else None),
        "evaluator_log_sha256": {case: loaded[case][1] for case in present},
        "holdout_touched": False,
    }
    (args.base / "paired-episodes.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in paired))
    (args.base / "paired-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
