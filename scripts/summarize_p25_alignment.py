"""Summarize P2.5 same-observation disagreement at the episode level."""

from __future__ import annotations

import argparse
import json
import random
import statistics as stats
from collections import defaultdict
from pathlib import Path


def mean(values):
    return sum(values) / len(values)


def bootstrap_interval(values, seed=20260927, samples=10000):
    rng = random.Random(seed)
    means = sorted(mean([values[rng.randrange(len(values))] for _ in values]) for _ in range(samples))
    return [means[int(samples * 0.025)], means[int(samples * 0.975)]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparisons", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.comparisons.read_text(encoding="utf-8").splitlines()]
    by_episode = defaultdict(list)
    for row in rows:
        by_episode[row["episode_serial"]].append(row)
    if len(by_episode) != 10:
        raise RuntimeError("Expected the preregistered 10 episodes")
    episodes = []
    for serial, group in sorted(by_episode.items()):
        group.sort(key=lambda row: row["query_in_episode"])
        if [row["query_in_episode"] for row in group] != list(range(len(group))):
            raise RuntimeError(f"Non-contiguous query indices for episode {serial}")
        third = max(1, len(group) // 3)
        early = mean([row["chunk_mse"] for row in group[:third]])
        late = mean([row["chunk_mse"] for row in group[-third:]])
        episodes.append({
            "episode_serial": serial,
            "success": group[0]["success"],
            "queries": len(group),
            "mean_chunk_mse": mean([row["chunk_mse"] for row in group]),
            "median_chunk_mse": stats.median(row["chunk_mse"] for row in group),
            "early_third_mse": early,
            "late_third_mse": late,
            "late_minus_early_mse": late - early,
            "mean_gripper_disagreement": mean([row["gripper_disagreement"] for row in group]),
            "first_gripper_disagreement_query": next(
                (row["query_in_episode"] for row in group if row["gripper_disagreement"] > 0), None
            ),
        })
    episode_means = [row["mean_chunk_mse"] for row in episodes]
    late_deltas = [row["late_minus_early_mse"] for row in episodes]
    per_step = [mean([row["rmse_per_step"][i] for row in rows]) for i in range(8)]
    per_dimension = [mean([row["rmse_per_dimension"][i] for row in rows]) for i in range(7)]
    student_margins = [margin for row in rows for margin in row["student_gripper_signed_margin"]]
    teacher_margins = [margin for row in rows for margin in row["teacher_gripper_signed_margin"]]
    result = {
        "schema_version": "1.0",
        "evidence_class": "diagnostic_development_resets_not_final",
        "queries": len(rows),
        "episode_count": len(episodes),
        "successes": sum(bool(row["success"]) for row in episodes),
        "episode_mean_chunk_mse_mean": mean(episode_means),
        "episode_mean_chunk_mse_median": stats.median(episode_means),
        "episode_mean_chunk_mse_bootstrap_95ci": bootstrap_interval(episode_means),
        "late_minus_early_mse_mean": mean(late_deltas),
        "late_minus_early_mse_median": stats.median(late_deltas),
        "late_minus_early_mse_bootstrap_95ci": bootstrap_interval(late_deltas),
        "episodes_late_worse": sum(value > 0 for value in late_deltas),
        "worst_episode_mean_chunk_mse": max(episode_means),
        "mean_rmse_by_executed_step": per_step,
        "mean_rmse_by_action_dimension": per_dimension,
        "mean_h17_token_mean_cosine": mean([row["student_teacher_h17_token_mean_cosine"] for row in rows]),
        "median_h17_token_mean_cosine": stats.median(row["student_teacher_h17_token_mean_cosine"] for row in rows),
        "student_gripper_near_threshold_fraction_abs_margin_lt_0_1": mean([abs(x) < 0.1 for x in student_margins]),
        "teacher_gripper_near_threshold_fraction_abs_margin_lt_0_1": mean([abs(x) < 0.1 for x in teacher_margins]),
        "episodes": episodes,
        "large_divergence_threshold": None,
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "queries", "successes", "episode_mean_chunk_mse_mean", "episodes_late_worse",
        "late_minus_early_mse_bootstrap_95ci", "mean_rmse_by_executed_step",
    )}))


if __name__ == "__main__":
    main()
