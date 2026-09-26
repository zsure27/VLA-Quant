"""Extract reproducible trajectory-level PEFT train or router-dev frames from LIBERO RLDS."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow_datasets as tfds

from qvla.inventory_peft_trajectories import stable_id


DEFAULT_POSITIONS = (0.1, 0.3, 0.5, 0.7, 0.9)


def decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return decode(value.item())
    return str(value)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chunk_actions(steps: list[dict[str, Any]], start: int, count: int = 8) -> np.ndarray:
    values = [np.asarray(step["action"], dtype=np.float32) for step in steps]
    selected = values[start:min(len(values), start + count)]
    selected.extend([selected[-1]] * (count - len(selected)))
    return np.stack(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-directory", required=True, type=Path)
    parser.add_argument("--trajectory-split", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=("peft_train", "router_dev"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--positions", default=",".join(str(value) for value in DEFAULT_POSITIONS))
    args = parser.parse_args()
    positions = tuple(float(value) for value in args.positions.split(","))
    if not positions or any(not 0 <= value <= 1 for value in positions) or len(set(positions)) != len(positions):
        raise ValueError("positions must be unique values in [0,1]")
    split = json.loads(args.trajectory_split.read_text(encoding="utf-8"))
    if split.get("dataset_version") != "1.0.0":
        raise RuntimeError("Unexpected trajectory split dataset version")
    declared = {row["dataset_order_index"]: row for row in split["episodes"]}
    builder = tfds.builder_from_directory(str(args.dataset_directory))
    dataset = builder.as_dataset(split="train", shuffle_files=False)
    args.output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for dataset_index, episode in enumerate(tfds.as_numpy(dataset)):
        row = declared.get(dataset_index)
        if row is None or row["role"] != args.role:
            continue
        steps = list(episode["steps"])
        if not steps:
            raise RuntimeError(f"Empty trajectory {dataset_index}")
        instruction = decode(steps[0]["language_instruction"])
        source_path = decode(episode["episode_metadata"]["file_path"])
        if stable_id(dataset_index, source_path, instruction) != row["trajectory_id"]:
            raise RuntimeError(f"Trajectory identity drift at dataset index {dataset_index}")
        selected_indices = sorted({int(round(position * (len(steps) - 1))) for position in positions})
        for step_index in selected_indices:
            step = steps[step_index]
            observation = step["observation"]
            filename = f"sample-{len(records):05d}.npz"
            destination = args.output / filename
            np.savez_compressed(
                destination,
                image=observation["image"],
                wrist_image=observation["wrist_image"],
                state=np.asarray(observation["state"], dtype=np.float32),
                action=chunk_actions(steps, step_index),
                instruction=np.asarray(instruction),
            )
            records.append({
                "file": filename,
                "sha256": digest(destination),
                "trajectory_id": row["trajectory_id"],
                "dataset_order_index": dataset_index,
                "source_file_path": source_path,
                "instruction": instruction,
                "step_index": step_index,
                "trajectory_steps": len(steps),
                "normalized_position": step_index / max(1, len(steps) - 1),
                "role": args.role,
            })
    expected_trajectories = sum(row["role"] == args.role for row in split["episodes"])
    found_trajectories = len({row["trajectory_id"] for row in records})
    if found_trajectories != expected_trajectories:
        raise RuntimeError(f"Expected {expected_trajectories} trajectories, found {found_trajectories}")
    manifest = {
        "schema_version": "1.0",
        "role": args.role,
        "holdout_touched": False,
        "selection": "fixed normalized temporal positions per complete trajectory; round(p*(steps-1)); duplicate indices removed",
        "requested_positions": positions,
        "trajectory_count": found_trajectories,
        "frame_count": len(records),
        "trajectory_split": str(args.trajectory_split),
        "trajectory_split_sha256": digest(args.trajectory_split),
        "samples": records,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "role": args.role, "trajectories": found_trajectories, "frames": len(records)}))


if __name__ == "__main__":
    main()

