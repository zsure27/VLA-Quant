"""Summarize trajectory-level router-dev action agreement against a fixed 12L baseline."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = q * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def load_metrics(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["sample"]: row for row in rows}


def aggregate(rows: list[dict]) -> dict:
    losses = [row["normalized_action"]["mse"] for row in rows]
    per_dim_mse = [statistics.mean(row["normalized_rmse_per_dim"][index] ** 2 for row in rows) for index in range(7)]
    teacher_margins = [abs(value) for row in rows for value in row["gripper_steps"]["teacher_signed_margin"]]
    candidate_margins = [abs(value) for row in rows for value in row["gripper_steps"]["candidate_signed_margin"]]
    return {
        "frames": len(rows),
        "mean_action_mse": statistics.mean(losses),
        "median_action_mse": statistics.median(losses),
        "p90_action_mse": percentile(losses, 0.9),
        "position_rmse": math.sqrt(statistics.mean(per_dim_mse[0:3])),
        "rotation_rmse": math.sqrt(statistics.mean(per_dim_mse[3:6])),
        "gripper_continuous_rmse": math.sqrt(per_dim_mse[6]),
        "gripper_disagreement": statistics.mean(row["raw_gripper_disagreement"] for row in rows),
        "teacher_gripper_abs_margin_mean": statistics.mean(teacher_margins),
        "candidate_gripper_abs_margin_mean": statistics.mean(candidate_margins),
        "candidate_gripper_abs_margin_p10": percentile(candidate_margins, 0.1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-manifest", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, action="append", help="NAME=metrics.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.sample_manifest.read_text(encoding="utf-8"))
    if manifest.get("role") != "router_dev" or manifest.get("holdout_touched") is not False:
        raise RuntimeError("Only the unsealed router_dev split is allowed")
    sample_rows = {row["file"]: row for row in manifest["samples"]}
    baseline = load_metrics(args.baseline)
    if set(baseline) != set(sample_rows):
        raise RuntimeError("Baseline coverage does not match router_dev manifest")
    candidates = {}
    for spec in args.candidate:
        name, raw_path = spec.split("=", 1)
        if name in candidates:
            raise ValueError(f"Duplicate candidate {name}")
        candidates[name] = load_metrics(Path(raw_path))
        if set(candidates[name]) != set(sample_rows):
            raise RuntimeError(f"Candidate coverage mismatch: {name}")
    baseline_by_trajectory: dict[str, list[float]] = {}
    for sample, row in baseline.items():
        baseline_by_trajectory.setdefault(sample_rows[sample]["trajectory_id"], []).append(row["normalized_action"]["mse"])
    baseline_trajectory_medians = {key: statistics.median(values) for key, values in baseline_by_trajectory.items()}
    output = {
        "schema_version": "1.0",
        "classification": "trajectory-level router_dev development evidence; offline_final_holdout untouched",
        "selection": manifest["selection"],
        "requested_positions": manifest["requested_positions"],
        "trajectory_count": manifest["trajectory_count"],
        "baseline": aggregate(list(baseline.values())),
        "candidates": {},
    }
    for name, metrics in candidates.items():
        candidate_by_trajectory: dict[str, list[float]] = {}
        for sample, row in metrics.items():
            candidate_by_trajectory.setdefault(sample_rows[sample]["trajectory_id"], []).append(row["normalized_action"]["mse"])
        candidate_medians = {key: statistics.median(values) for key, values in candidate_by_trajectory.items()}
        deltas = [candidate_medians[key] - baseline_trajectory_medians[key] for key in sorted(baseline_trajectory_medians)]
        worst_count = max(1, math.ceil(len(deltas) * 0.1))
        summary = aggregate(list(metrics.values()))
        summary.update({
            "per_trajectory_median_mean": statistics.mean(candidate_medians.values()),
            "per_trajectory_median_median": statistics.median(candidate_medians.values()),
            "fraction_trajectories_improved_vs_12l": sum(value < 0 for value in deltas) / len(deltas),
            "median_trajectory_delta_vs_12l": statistics.median(deltas),
            "worst_decile_mean_trajectory_delta_vs_12l": statistics.mean(sorted(deltas, reverse=True)[:worst_count]),
            "max_trajectory_regression_vs_12l": max(deltas),
        })
        output["candidates"][name] = summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: value for name, value in output["candidates"].items()}, sort_keys=True))


if __name__ == "__main__":
    main()

