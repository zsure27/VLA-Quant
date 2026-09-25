"""Inventory LIBERO RLDS episodes and create deterministic trajectory splits."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds


def stable_id(dataset_index: int, file_path: str, instruction: str) -> str:
    payload = f"vla-peft-split-v1\0{dataset_index}\0{file_path}\0{instruction}"
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    builder = tfds.builder_from_directory(str(args.dataset_directory))
    dataset = builder.as_dataset(split="train", shuffle_files=False)
    episodes = []
    action_min = np.full(7, np.inf)
    action_max = np.full(7, -np.inf)
    gripper_values = set()
    standardized_gripper_values = set()
    for dataset_index, episode in enumerate(tfds.as_numpy(dataset)):
        steps = list(episode["steps"])
        if not steps:
            raise RuntimeError("empty trajectory")
        path = episode["episode_metadata"]["file_path"].decode("utf-8")
        instructions = {step["language_instruction"].decode("utf-8") for step in steps}
        if len(instructions) != 1:
            raise RuntimeError(f"instruction changes inside trajectory: {path}")
        action = np.stack([step["action"] for step in steps])
        state = np.stack([step["observation"]["state"] for step in steps])
        if action.shape[1:] != (7,) or state.shape[1:] != (8,):
            raise RuntimeError(f"field shape mismatch: {path}")
        if not np.isfinite(action).all() or not np.isfinite(state).all():
            raise RuntimeError(f"non-finite trajectory: {path}")
        action_min = np.minimum(action_min, action.min(axis=0))
        action_max = np.maximum(action_max, action.max(axis=0))
        gripper_values.update(float(x) for x in np.unique(action[:, 6]))
        standardized_gripper_values.update(float(x) for x in np.unique(1.0 - np.clip(action[:, 6], 0.0, 1.0)))
        instruction = next(iter(instructions))
        episodes.append({"trajectory_id": stable_id(dataset_index, path, instruction),
                         "dataset_order_index": dataset_index, "source_file_path": path,
                         "instruction": instruction, "steps": len(steps)})

    by_instruction = defaultdict(list)
    for row in episodes:
        by_instruction[row["instruction"]].append(row)
    split_counts = defaultdict(int)
    for rows in by_instruction.values():
        rows.sort(key=lambda row: row["trajectory_id"])
        n = len(rows)
        train_end = int(n * 0.70)
        dev_end = train_end + int(n * 0.15)
        for index, row in enumerate(rows):
            row["role"] = "peft_train" if index < train_end else ("router_dev" if index < dev_end else "offline_final_holdout")
            split_counts[row["role"]] += 1
    episodes.sort(key=lambda row: row["trajectory_id"])
    ids = [row["trajectory_id"] for row in episodes]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate trajectory identity")
    manifest = {
        "schema_version": "1.0", "dataset": builder.info.name,
        "dataset_version": str(builder.info.version), "trajectory_count": len(episodes),
        "transition_count": sum(row["steps"] for row in episodes),
        "split_method": "within each instruction, sort SHA256(vla-peft-split-v1\\0 + deterministic dataset index + original file_path + instruction), then 70/15/remaining",
        "split_counts": dict(sorted(split_counts.items())),
        "fields": {"images": ["observation.image uint8[256,256,3]", "observation.wrist_image uint8[256,256,3]"],
                   "instruction": "language_instruction UTF-8, constant within trajectory",
                   "proprio": "observation.state float32[8]",
                   "action": "raw float32[7]; LIBERO transform maps raw gripper -1=open,+1=close via 1-clip(x,0,1) to 1=open,0=close; dimensions 0:6 use bounds_q99 while gripper stays unnormalized",
                   "action_chunk": "8 steps (OpenVLA-OFT runtime constant)"},
        "action_min": action_min.tolist(), "action_max": action_max.tolist(),
        "gripper_unique_values": sorted(gripper_values),
        "standardized_gripper_unique_values": sorted(standardized_gripper_values),
        "instructions": {key: len(value) for key, value in sorted(by_instruction.items())},
        "leakage_policy": {"unit": "trajectory", "frame_random_split": False,
                           "spatial_states_0_49": "historical/development closed-loop evidence only; never training trajectories or blind test",
                           "offline_final_holdout": "sealed until PEFT type, modules, rank, experts and router features are frozen",
                           "closed_loop_final": "use changed simulator reset/seed or preregistered controlled perturbations; record initial observation hashes"},
        "episodes": episodes,
    }
    (args.output / "trajectory_split.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: value for key, value in manifest.items() if key != "episodes"}
    (args.output / "inventory_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "trajectory_count": len(episodes), "transition_count": manifest["transition_count"],
                      "split_counts": manifest["split_counts"], "instructions": len(by_instruction)}, sort_keys=True))


if __name__ == "__main__":
    main()
