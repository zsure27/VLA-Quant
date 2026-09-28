"""Compare frozen LoRA/BF16 disagreement across P2.5 observation distributions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics as stats


def mean(values):
    return sum(values) / len(values)


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def offline_distribution(candidate_file: Path, baseline_file: Path, manifest_file: Path, expected: int) -> dict:
    candidate = json.loads(candidate_file.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if len(candidate) != expected or len(baseline) != expected or len(manifest["samples"]) < expected:
        raise RuntimeError("Unexpected frame count")
    names = [row["file"] for row in manifest["samples"][:expected]]
    if names != [row["sample"] for row in candidate] or names != [row["sample"] for row in baseline]:
        raise RuntimeError("Metric pairing mismatch")
    trajectories = {}
    for candidate_row, baseline_row, metadata in zip(candidate, baseline, manifest["samples"][:expected]):
        trajectory = metadata["trajectory_id"]
        trajectories.setdefault(trajectory, []).append(candidate_row["raw_action"]["mse"])
    mse = [row["raw_action"]["mse"] for row in candidate]
    baseline_mse = [row["raw_action"]["mse"] for row in baseline]
    gripper = [row["raw_gripper_disagreement"] for row in candidate]
    baseline_gripper = [row["raw_gripper_disagreement"] for row in baseline]
    return {
        "frames": expected, "trajectories": len(trajectories),
        "lora_raw_mse_mean": mean(mse), "lora_raw_mse_median": stats.median(mse),
        "lora_raw_mse_worst_decile_threshold": percentile(mse, 0.9),
        "per_trajectory_median_raw_mse": stats.median(mean(values) for values in trajectories.values()),
        "baseline_raw_mse_mean": mean(baseline_mse),
        "lora_minus_baseline_raw_mse": mean([a - b for a, b in zip(mse, baseline_mse)]),
        "frames_lora_improved_fraction": mean([a < b for a, b in zip(mse, baseline_mse)]),
        "lora_gripper_disagreement_mean": mean(gripper),
        "baseline_gripper_disagreement_mean": mean(baseline_gripper),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True, type=Path)
    parser.add_argument("--router-dir", required=True, type=Path)
    parser.add_argument("--student-analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    train = offline_distribution(args.train_dir / "lora-metrics.json",
                                 args.train_dir / "exact12l-metrics.json",
                                 args.train_dir / "peft-train-frames-manifest.json", 80)
    dev = offline_distribution(args.router_dir / "D-metrics.json",
                               args.router_dir / "exact12l-baseline-metrics.json",
                               args.router_dir / "router-dev-sample-manifest.json", 295)
    student = json.loads(args.student_analysis.read_text(encoding="utf-8"))
    if student["episodes"] != 500 or student["successes"] != 408:
        raise RuntimeError("Unexpected student-visited source")
    result = {
        "schema_version": "1.0", "classification": "P2.5 observational distribution comparison",
        "peft_train_other_frame": train, "router_dev": dev,
        "student_visited": {
            "policy_queries": student["queries"], "episodes": student["episodes"],
            "successes": student["successes"],
            "lora_raw_mse_query_mean": student["query_mean_chunk_mse"],
            "lora_raw_mse_query_median": student["query_median_chunk_mse"],
            "lora_gripper_disagreement_query_mean": student["query_mean_gripper_disagreement"],
            "success_episode_mean_mse": student["success_episode_mean_mse"],
            "failure_episode_mean_mse": student["failure_episode_mean_mse"],
        },
        "limitations": [
            "Three distributions have different frame/trajectory sampling and query weighting; raw means are descriptive, not a causal covariate-shift proof.",
            "Student-visited data are induced by the LoRA policy; no full 500-episode no-LoRA same-state query control is included.",
            "The peft_train comparison uses other frames from training-role trajectories, not the exact 80 optimizer frames.",
            "Success and same-state teacher error can share task difficulty or episode length as confounders.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"train": train, "router_dev": dev, "student_visited": result["student_visited"]}))


if __name__ == "__main__":
    main()
