"""Select the fourth causal policy query from each completed student episode."""
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
    queries = [row for row in events if row["record_type"] == "query" and row["query_in_episode"] == 3]
    if len(queries) != 100 or {row["episode_serial"] for row in queries} != set(range(100)):
        raise RuntimeError("Expected exactly one causal query 3 for each of 100 episodes")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = []
    for index, row in enumerate(sorted(queries, key=lambda item: item["episode_serial"])):
        source = args.student_run / row["file"]
        if file_sha256(source) != row["file_sha256"]:
            raise RuntimeError(f"Capture SHA mismatch: {source}")
        with np.load(source, allow_pickle=False) as sample:
            image, wrist_image = sample["image"], sample["wrist_image"]
            state = sample["state"].astype(np.float32)
            instruction = str(sample["instruction"].item())
            action = sample["student_action"].astype(np.float32)
        if observation_hash(image, wrist_image, state, instruction) != row["observation_sha256"]:
            raise RuntimeError(f"Observation hash mismatch: {source}")
        target = args.output / f"sample-{index:05d}.npz"
        np.savez_compressed(target, image=image, wrist_image=wrist_image, state=state,
                            instruction=np.array(instruction), action=action[0])
        manifest.append({"file": target.name, "source": row["file"], "source_sha256": row["file_sha256"],
                         "observation_sha256": row["observation_sha256"],
                         "episode_serial": row["episode_serial"], "query_in_episode": 3,
                         "sample_sha256": file_sha256(target)})
    (args.output / "manifest.json").write_text(json.dumps({
        "schema_version": "1.0", "role": "student_visited", "selection": "query_in_episode=3, executed_step_start=24",
        "count": len(manifest), "samples": manifest,
        "dataset_action_role": "placeholder student first step; not a teacher target",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
