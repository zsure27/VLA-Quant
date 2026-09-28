"""Diagnose control-time alignment on already captured P2.5 student observations."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--baseline-metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    original = args.capture / "20260927-107-p25-on-policy-alignment"
    student = original / "student-rollout"
    teacher = original / "bf16-same-observation"
    comparisons = rows(teacher / "same-observation-comparisons.jsonl")
    baseline = json.loads(args.baseline_metrics.read_text(encoding="utf-8"))
    events = rows(student / "on-policy-events.jsonl")
    query_events = [row for row in events if row["record_type"] == "query"]
    outcomes = {row["episode_serial"]: row for row in events if row["record_type"] == "episode_end"}
    if len(comparisons) != len(baseline) or len(baseline) != len(query_events) or len(comparisons) != 140:
        raise RuntimeError("Expected 140 paired, ordered policy queries")
    trajectory = defaultdict(list)
    per_step_sq = np.zeros((8, 7), dtype=np.float64)
    per_step_gripper = np.zeros(8, dtype=np.int64)
    for index, (comparison, base, event) in enumerate(zip(comparisons, baseline, query_events)):
        if event["file"] != comparison["file"] or base["sample"] != f"sample-{index:06d}.npz":
            raise RuntimeError(f"Pairing mismatch at query {index}")
        with np.load(student / event["file"], allow_pickle=False) as sample:
            student_action = sample["student_action"].astype(np.float64)
            proprio = sample["state"].astype(np.float64)
        with np.load(teacher / comparison["teacher_response_file"], allow_pickle=False) as sample:
            teacher_action = sample["teacher_action"].astype(np.float64)
        if student_action.shape != (8, 7) or teacher_action.shape != (8, 7):
            raise RuntimeError("Expected executed 8x7 action chunks")
        difference = student_action - teacher_action
        per_step_sq += difference ** 2
        # OFT binarizes the model's [0, 1] gripper at 0.5 before sign inversion.
        disagree = (student_action[:, 6] >= 0.5) != (teacher_action[:, 6] >= 0.5)
        per_step_gripper += disagree.astype(np.int64)
        base_gripper = np.asarray(base["gripper_steps"]["candidate_signed_margin"])
        teacher_gripper = np.asarray(base["gripper_steps"]["teacher_signed_margin"])
        baseline_disagree = np.any(np.sign(base_gripper) != np.sign(teacher_gripper))
        trajectory[event["episode_serial"]].append({
            "query": event["query_in_episode"], "executed_step_start": event["executed_step_start"],
            "proprio": proprio.tolist(), "lora_action_mse": float(np.mean(difference ** 2)),
            "baseline_action_mse": float(base["raw_action"]["mse"]),
            "position_rmse": float(np.sqrt(np.mean(difference[:, :3] ** 2))),
            "rotation_rmse": float(np.sqrt(np.mean(difference[:, 3:6] ** 2))),
            "gripper_rmse": float(np.sqrt(np.mean(difference[:, 6] ** 2))),
            "lora_gripper_disagreement": bool(np.any(disagree)),
            "baseline_gripper_disagreement": bool(baseline_disagree),
            "lora_minus_baseline_mse": float(np.mean(difference ** 2) - base["raw_action"]["mse"]),
        })
    episodes = []
    for episode, timeline in sorted(trajectory.items()):
        success = bool(outcomes[episode]["success"])
        episodes.append({"episode_serial": episode, "success": success, "queries": len(timeline),
                         "first_lora_gripper_disagreement_query": next((r["query"] for r in timeline if r["lora_gripper_disagreement"]), None),
                         "first_baseline_gripper_disagreement_query": next((r["query"] for r in timeline if r["baseline_gripper_disagreement"]), None),
                         "mean_lora_minus_baseline_mse": float(np.mean([r["lora_minus_baseline_mse"] for r in timeline])),
                         "timeline": timeline})
    output = {"classification": "retrospective P2.5 mechanism diagnosis; development resets",
              "queries": len(comparisons), "episodes": episodes,
              "per_chunk_index": [{"index": i, "position_rmse": float(np.sqrt(np.mean(per_step_sq[i, :3] / 140))),
                                   "rotation_rmse": float(np.sqrt(np.mean(per_step_sq[i, 3:6] / 140))),
                                   "gripper_rmse": float(np.sqrt(per_step_sq[i, 6] / 140),
                                   ), "gripper_disagreement_rate": float(per_step_gripper[i] / 140)} for i in range(8)],
              "limitations": ["No task-stage/failure-onset annotation is present in captured events.",
                              "The no-LoRA actions are paired on LoRA-visited observations, not a no-LoRA rollout.",
                              "A first-large-divergence threshold was not preregistered; no large-divergence event is claimed."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"queries": len(comparisons), "episodes": len(episodes),
                      "failure_episodes": [e["episode_serial"] for e in episodes if not e["success"]],
                      "per_chunk_index": output["per_chunk_index"]}))


if __name__ == "__main__":
    main()
