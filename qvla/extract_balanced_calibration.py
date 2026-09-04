from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow_datasets as tfds


SUITES = (
    "libero_spatial_no_noops",
    "libero_object_no_noops",
    "libero_goal_no_noops",
    "libero_10_no_noops",
)


def decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return decode_text(value.item())
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--per-suite", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.per_suite < 1:
        raise ValueError("per-suite must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    suite_records: dict[str, list[dict[str, Any]]] = {}

    for suite_index, suite in enumerate(SUITES):
        dataset_dir = args.data_root / suite / "1.0.0"
        builder = tfds.builder_from_directory(str(dataset_dir))
        total = int(builder.info.splits["train"].num_examples)
        if total < args.per_suite:
            raise RuntimeError(f"{suite} has {total} episodes, need {args.per_suite}")
        selected = set(int(x) for x in rng.choice(total, args.per_suite, replace=False))
        dataset = builder.as_dataset(split="train", shuffle_files=False)
        records = []

        for episode_index, episode in enumerate(tfds.as_numpy(dataset)):
            if episode_index not in selected:
                continue
            steps = list(episode["steps"])
            if not steps:
                raise RuntimeError(f"Empty episode: {suite}/{episode_index}")
            low = max(0, int(len(steps) * 0.1))
            high = max(low + 1, int(len(steps) * 0.9))
            step_index = int(rng.integers(low, min(high, len(steps))))
            step = steps[step_index]
            observation = step["observation"]
            records.append({
                "suite": suite,
                "episode": episode_index,
                "step": step_index,
                "steps": len(steps),
                "instruction": decode_text(step["language_instruction"]),
                "image": observation["image"],
                "wrist_image": observation["wrist_image"],
                "state": observation["state"].astype(np.float32),
                "action": step["action"].astype(np.float32),
            })
        if len(records) != args.per_suite:
            raise RuntimeError(f"Extracted {len(records)} from {suite}, expected {args.per_suite}")
        rng.shuffle(records)
        suite_records[suite] = records
        print(suite, len(records), "/", total)

    manifest = {"seed": args.seed, "per_suite": args.per_suite, "samples": []}
    sample_index = 0
    for offset in range(args.per_suite):
        for suite in SUITES:
            record = suite_records[suite][offset]
            filename = f"sample-{sample_index:04d}.npz"
            destination = args.output_dir / filename
            np.savez_compressed(
                destination,
                image=record["image"],
                wrist_image=record["wrist_image"],
                state=record["state"],
                action=record["action"],
                instruction=np.asarray(record["instruction"]),
            )
            manifest["samples"].append({
                "index": sample_index,
                "file": filename,
                "suite": suite,
                "episode": record["episode"],
                "step": record["step"],
                "steps": record["steps"],
                "instruction": record["instruction"],
            })
            sample_index += 1

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("samples:", sample_index)
    print("manifest:", manifest_path)
    print("BALANCED CALIBRATION EXTRACTION: PASS")


if __name__ == "__main__":
    main()
