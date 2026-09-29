"""Aggregate completed A1 shards after strict task/reset and manifest checks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


CASES = ("BF16", "C0", "C1", "C2", "C3")
IDENTITY = ("task_id", "init_state_index", "model_seed", "env_seed", "init_state_sha256")
SHARDS = ("05-09", "10-19", "20-29", "30-34", "35-39", "40-49")


def exact_mcnemar(rescue: int, new_failure: int) -> float:
    n = rescue + new_failure
    if n == 0:
        return 1.0
    low = min(rescue, new_failure)
    combination = 1
    cumulative = 1
    for k in range(1, low + 1):
        combination = combination * (n - k + 1) // k
        cumulative += combination
    return min(1.0, 2.0 * cumulative / 2**n)


def stratified_interval(rows: list[dict], trials: int = 20000) -> list[float]:
    groups = [[int(row["C3"]) - int(row["C0"]) for row in rows if row["task_id"] == task]
              for task in range(10)]
    rng = random.Random(20260929)
    values = [sum(rng.choice(group) for group in groups for _ in group) / len(rows)
              for _ in range(trials)]
    values.sort()
    return [values[int(0.025 * trials)], values[int(0.975 * trials)]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--require-full", action="store_true")
    args = parser.parse_args()
    rows_by_key: dict[tuple[int, int], dict] = {}
    included: list[str] = []
    for shard in SHARDS:
        path = args.results / f"paired-{shard}-episodes.jsonl"
        if not path.is_file():
            if args.require_full:
                raise ValueError(f"missing preregistered shard: {shard}")
            continue
        included.append(shard)
        low, high = map(int, shard.split("-"))
        expected = {(task, reset) for task in range(10) for reset in range(low, high + 1)}
        seen: set[tuple[int, int]] = set()
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if any(field not in row for field in (*IDENTITY, *CASES)):
                raise ValueError(f"missing fields in {path}")
            key = (row["task_id"], row["init_state_index"])
            if key not in expected or key in seen or key in rows_by_key:
                raise ValueError(f"unexpected or duplicate task/reset {key} in {path}")
            seen.add(key)
            rows_by_key[key] = row
        if seen != expected:
            raise ValueError(f"incomplete shard {shard}: {len(seen)}/{len(expected)}")
    if not rows_by_key:
        raise ValueError("no completed shards")
    rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    successes = {case: sum(bool(row[case]) for row in rows) for case in CASES}
    by_task = {str(task): {case: sum(bool(row[case]) for row in rows if row["task_id"] == task)
                           for case in CASES} for task in range(10)}
    task_delta = {task: result["C3"] - result["C0"] for task, result in by_task.items()}
    rescue = sum(not row["C0"] and row["C3"] for row in rows)
    new_failure = sum(row["C0"] and not row["C3"] for row in rows)
    headroom = successes["C2"] - successes["C0"]
    interval = stratified_interval(rows)
    complete = len(included) == len(SHARDS)
    output = {
        "classification": "same official initial states, different environment random stream; not blind holdout",
        "shards": included,
        "full_preregistered_coverage": complete,
        "episodes_per_case": len(rows),
        "successes": successes,
        "by_task": by_task,
        "task_C3_minus_C0": task_delta,
        "tasks_net_improved": sum(delta > 0 for delta in task_delta.values()),
        "tasks_net_harmed": sum(delta < 0 for delta in task_delta.values()),
        "C3_minus_C0": successes["C3"] - successes["C0"],
        "rescue": rescue,
        "new_failure": new_failure,
        "exact_mcnemar_p": exact_mcnemar(rescue, new_failure),
        "paired_task_stratified_bootstrap_95ci_fraction": interval,
        "C2_minus_C0_headroom": headroom,
        "W4_recovery_fraction_descriptive_only": (successes["C3"] - successes["C0"]) / headroom
        if headroom > 0 else None,
        "A1_support_gate": bool(complete and successes["C3"] > successes["C0"]
                                and interval[0] > 0
                                and sum(delta > 0 for delta in task_delta.values()) >= 2),
        "holdout_touched": False,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
