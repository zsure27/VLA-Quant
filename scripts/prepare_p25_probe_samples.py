"""Convert hash-verified student-visited observations to probe's sample contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qvla.on_policy_capture import file_sha256, observation_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    events = [json.loads(line) for line in (args.student_run / "on-policy-events.jsonl").read_text().splitlines()]
    queries = [row for row in events if row["record_type"] == "query"]
    if len(queries) != 140 or args.output.exists():
        raise RuntimeError("Expected exactly 140 captured queries and a fresh output directory")
    args.output.mkdir(parents=True)
    manifest = []
    for index, row in enumerate(queries):
        source = args.student_run / row["file"]
        if file_sha256(source) != row["file_sha256"]:
            raise RuntimeError(f"Source hash mismatch: {source}")
        with np.load(source, allow_pickle=False) as sample:
            image = sample["image"]
            wrist_image = sample["wrist_image"]
            state = sample["state"].astype(np.float32)
            instruction = str(sample["instruction"].item())
            student_action = sample["student_action"].astype(np.float32)
        if observation_hash(image, wrist_image, state, instruction) != row["observation_sha256"]:
            raise RuntimeError(f"Observation hash mismatch: {source}")
        if student_action.shape != (8, 7):
            raise RuntimeError(f"Unexpected action shape: {student_action.shape}")
        target = args.output / f"sample-{index:06d}.npz"
        np.savez_compressed(target, image=image, wrist_image=wrist_image, state=state,
                            instruction=np.array(instruction), action=student_action[0])
        manifest.append({"sample": target.name, "source": row["file"],
                         "source_sha256": row["file_sha256"], "observation_sha256": row["observation_sha256"],
                         "episode_serial": row["episode_serial"], "probe_sha256": file_sha256(target)})
    (args.output / "manifest.json").write_text(json.dumps({"schema_version": "1.0", "role": "student_visited",
        "count": len(manifest), "samples": manifest, "dataset_action_role": "placeholder_student_first_step; never used as teacher target"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
