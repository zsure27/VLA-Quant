"""Validate and summarize a preregistered paired P2C closed-loop pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

from diagnostics.baseline_shard_analysis import parse_episodes


CASES = {"C0": "C0-12L", "C1": "C1-12L-scale", "C2": "C2-14L"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(min(left_only, right_only) + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round(probability * (len(ordered) - 1))]


def paired_matrix(rows: list[dict], left: str, right: str) -> dict:
    result = {"both_success": 0, f"{left}_only": 0, f"{right}_only": 0, "both_fail": 0}
    for row in rows:
        a, b = row[left], row[right]
        key = "both_success" if a and b else f"{left}_only" if a else f"{right}_only" if b else "both_fail"
        result[key] += 1
    result["mcnemar_exact_p"] = exact_mcnemar(result[f"{left}_only"], result[f"{right}_only"])
    return result


def bootstrap(rows: list[dict], iterations: int = 20000) -> dict:
    rng = random.Random(7)
    deltas, recoveries = [], []
    for _ in range(iterations):
        sampled = [rows[rng.randrange(len(rows))] for _ in rows]
        rates = {case: sum(row[case] for row in sampled) / len(sampled) for case in CASES}
        deltas.append(rates["C1"] - rates["C0"])
        denominator = rates["C2"] - rates["C0"]
        if denominator > 0:
            recoveries.append((rates["C1"] - rates["C0"]) / denominator)
    return {
        "method": "paired episode bootstrap, seed=7",
        "iterations": iterations,
        "scale_minus_12l_95ci": [percentile(deltas, 0.025), percentile(deltas, 0.975)],
        "recovery_fraction_95ci_when_denominator_positive": [
            percentile(recoveries, 0.025), percentile(recoveries, 0.975)
        ],
        "recovery_fraction_valid_draws": len(recoveries),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    protocol = json.loads((args.directory / "preregistered-protocol.json").read_text())
    offset, count = protocol["initial_state_offset"], protocol["trials_per_task"]
    expected = {(task, state) for task in range(protocol["task_count"]) for state in range(offset, offset + count)}
    loaded = {}
    evidence = {}
    for case, name in CASES.items():
        directory = args.directory / name
        logs = list(directory.glob("EVAL-*.txt"))
        if len(logs) != 1 or (directory / "exit-code.txt").read_text().strip() != "0":
            raise ValueError(f"{case} is incomplete")
        rows = parse_episodes(logs[0].read_text())
        if {(row["task_id"], row["init_state_index"]) for row in rows} != expected:
            raise ValueError(f"{case} coverage mismatch")
        loaded[case] = {(row["task_id"], row["init_state_index"]): row for row in rows}
        evidence[case] = {"eval_log": str(logs[0]), "sha256": digest(logs[0])}
    paired = []
    for identity in sorted(expected):
        source = loaded["C0"][identity]
        for case in ("C1", "C2"):
            peer = loaded[case][identity]
            for field in ("task_id", "init_state_index", "model_seed", "env_seed", "init_state_sha256"):
                if peer[field] != source[field]:
                    raise ValueError(f"paired manifest mismatch for {case} {identity} {field}")
        paired.append({
            **{field: source[field] for field in ("task_id", "init_state_index", "model_seed", "env_seed", "init_state_sha256")},
            **{case: loaded[case][identity]["success"] for case in CASES},
            **{f"{case}_episode_errors": loaded[case][identity]["episode_errors"] for case in CASES},
        })
    rates = {case: sum(row[case] for row in paired) / len(paired) for case in CASES}
    denominator = rates["C2"] - rates["C0"]
    summary = {
        "schema_version": "1.0",
        "classification": protocol["classification"],
        "episodes_per_config": len(paired),
        "successes": {case: sum(row[case] for row in paired) for case in CASES},
        "success_rates": rates,
        "recovery_fraction": (rates["C1"] - rates["C0"]) / denominator if denominator > 0 else None,
        "recovery_fraction_denominator_positive": denominator > 0,
        "paired_C0_C1": paired_matrix(paired, "C0", "C1"),
        "paired_C1_C2": paired_matrix(paired, "C1", "C2"),
        "bootstrap": bootstrap(paired),
        "evidence": evidence,
        "protocol": protocol,
    }
    (args.directory / "paired-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.directory / "paired-episodes.jsonl").open("w") as stream:
        for row in paired:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    with (args.directory / "per-task.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task_id", "episodes", "C0_successes", "C1_successes", "C2_successes"])
        for task in range(protocol["task_count"]):
            subset = [row for row in paired if row["task_id"] == task]
            writer.writerow([task, len(subset), *(sum(row[case] for row in subset) for case in CASES)])
    print(json.dumps({key: summary[key] for key in ("successes", "success_rates", "recovery_fraction", "paired_C0_C1", "paired_C1_C2")}, sort_keys=True))


if __name__ == "__main__":
    main()
