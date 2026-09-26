"""Query the frozen BF16 policy on observations actually visited by a student."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model
from experiments.robot.openvla_utils import get_action

from qvla.on_policy_capture import H17SummaryRecorder, action_metrics, file_sha256, observation_hash
from qvla.reproducibility import seed_all


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else float("nan")


def load_events(path: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    queries: list[dict[str, Any]] = []
    outcomes: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["record_type"] == "query":
            queries.append(row)
        elif row["record_type"] == "episode_end":
            outcomes[row["episode_serial"]] = row
    if not queries:
        raise RuntimeError("No on-policy queries found")
    missing = sorted({row["episode_serial"] for row in queries} - set(outcomes))
    if missing:
        raise RuntimeError(f"Missing episode outcomes: {missing}")
    return queries, outcomes


def initialize(checkpoint: str, seed: int):
    seed_all(seed, tensorflow=True)
    cfg = GenerateConfig(
        pretrained_checkpoint=checkpoint,
        task_suite_name="libero_spatial",
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=True,
        seed=seed,
    )
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    model.language_model.config.use_cache = False
    for module in (model, action_head, proprio_projector):
        module.requires_grad_(False)
        module.eval()
    model.vision_backbone.set_num_images_in_input(2)
    return cfg, model, action_head, proprio_projector, noisy_action_projector, processor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--student-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--large-disagreement-threshold", type=float)
    args = parser.parse_args()
    if args.large_disagreement_threshold is not None and args.large_disagreement_threshold <= 0:
        raise ValueError("large-disagreement-threshold must be positive")
    queries, outcomes = load_events(args.student_run / "on-policy-events.jsonl")
    args.output.mkdir(parents=True, exist_ok=False)
    response_dir = args.output / "teacher-responses"
    response_dir.mkdir()
    cfg, model, action_head, proprio_projector, noisy_action_projector, processor = initialize(
        args.checkpoint, args.seed)
    h17 = H17SummaryRecorder(model)
    comparison_path = args.output / "same-observation-comparisons.jsonl"
    comparisons: list[dict[str, Any]] = []
    try:
        with comparison_path.open("x", encoding="utf-8") as stream:
            for index, query in enumerate(queries):
                sample_path = args.student_run / query["file"]
                if file_sha256(sample_path) != query["file_sha256"]:
                    raise RuntimeError(f"Captured sample hash mismatch: {sample_path}")
                with np.load(sample_path, allow_pickle=False) as sample:
                    image = sample["image"]
                    wrist_image = sample["wrist_image"]
                    state = sample["state"].astype(np.float32)
                    instruction = str(sample["instruction"].item())
                    student_action = sample["student_action"].astype(np.float32)
                    student_h17_mean = sample["h17_token_mean"].astype(np.float32)
                    student_h17_last = sample["h17_last_token"].astype(np.float32)
                if observation_hash(image, wrist_image, state, instruction) != query["observation_sha256"]:
                    raise RuntimeError(f"Observation hash mismatch: {sample_path}")
                observation = {"full_image": image, "wrist_image": wrist_image, "state": state}
                h17.reset()
                seed_all(args.seed, tensorflow=False)
                teacher_action = np.asarray(get_action(
                    cfg,
                    model,
                    observation,
                    instruction,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                ), dtype=np.float32)
                teacher_h17 = h17.take()
                response_path = response_dir / f"query-{index:06d}.npz"
                np.savez_compressed(
                    response_path,
                    teacher_action=teacher_action,
                    teacher_h17_token_mean=teacher_h17["h17_token_mean"],
                    teacher_h17_last_token=teacher_h17["h17_last_token"],
                )
                metrics = action_metrics(student_action, teacher_action)
                outcome = outcomes[query["episode_serial"]]
                row = {
                    **query,
                    "success": bool(outcome["success"]),
                    "episode_aborted": bool(outcome["aborted"]),
                    "teacher_response_file": str(response_path.relative_to(args.output)),
                    "teacher_response_sha256": file_sha256(response_path),
                    "student_teacher_h17_token_mean_cosine": cosine(student_h17_mean, teacher_h17["h17_token_mean"]),
                    "student_teacher_h17_last_token_cosine": cosine(student_h17_last, teacher_h17["h17_last_token"]),
                    **metrics,
                }
                comparisons.append(row)
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
    finally:
        h17.close()
    by_episode: dict[int, list[dict[str, Any]]] = {}
    for row in comparisons:
        by_episode.setdefault(row["episode_serial"], []).append(row)
    episodes = []
    for episode, rows in sorted(by_episode.items()):
        threshold = args.large_disagreement_threshold
        first_large = next((row["query_in_episode"] for row in rows if threshold is not None and row["chunk_rmse"] >= threshold), None)
        first_gripper = next((row["query_in_episode"] for row in rows if row["gripper_disagreement"] > 0), None)
        episodes.append({
            "episode_serial": episode,
            "task": rows[0]["task"],
            "success": rows[0]["success"],
            "queries": len(rows),
            "mean_chunk_mse": float(np.mean([row["chunk_mse"] for row in rows])),
            "max_chunk_rmse": float(np.max([row["chunk_rmse"] for row in rows])),
            "first_large_policy_divergence_query": first_large,
            "first_large_policy_divergence_threshold": threshold,
            "first_gripper_divergence_query": first_gripper,
        })
    summary = {
        "schema_version": "1.0",
        "question": "Does rank8 Recovery-LoRA remain close to BF16 on observations induced by its own actions?",
        "distribution": "student-visited closed-loop observations",
        "teacher": "frozen BF16 queried on exactly the captured student observation",
        "future_outcome_used_as_input": False,
        "action_chunk_execution": "all 8 predicted actions execute open-loop before re-inference",
        "queries": len(comparisons),
        "episodes": len(episodes),
        "successful_episodes": sum(row["success"] for row in episodes),
        "mean_chunk_mse": float(np.mean([row["chunk_mse"] for row in comparisons])),
        "median_chunk_mse": float(np.median([row["chunk_mse"] for row in comparisons])),
        "mean_position_rmse": float(np.mean([row["position_rmse"] for row in comparisons])),
        "mean_rotation_rmse": float(np.mean([row["rotation_rmse"] for row in comparisons])),
        "mean_gripper_continuous_rmse": float(np.mean([row["gripper_continuous_rmse"] for row in comparisons])),
        "mean_gripper_disagreement": float(np.mean([row["gripper_disagreement"] for row in comparisons])),
        "large_disagreement_threshold": args.large_disagreement_threshold,
        "threshold_note": "null until preregistered from router_dev; no post-hoc large-divergence classification" if args.large_disagreement_threshold is None else "preregistered before this teacher query",
        "per_episode": episodes,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "checkpoint": args.checkpoint,
        "checkpoint_config_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(args.checkpoint).glob("*.json"))
        },
        "student_events_sha256": file_sha256(args.student_run / "on-policy-events.jsonl"),
        "comparison_sha256": file_sha256(comparison_path),
        "seed": args.seed,
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("queries", "episodes", "successful_episodes", "mean_chunk_mse", "mean_gripper_disagreement")}))


if __name__ == "__main__":
    main()
