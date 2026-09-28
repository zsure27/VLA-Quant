"""Hash-verified early student observations for two fixed development tasks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qvla.on_policy_capture import file_sha256, observation_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-shard", required=True, type=Path)
    parser.add_argument("--student-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    pairs = [json.loads(line) for line in (args.paired_shard / "paired-episodes.jsonl").read_text().splitlines()]
    assert len(pairs) == 100
    chosen = {index for index, pair in enumerate(pairs) if pair["task_id"] in (1, 2)}
    assert len(chosen) == 20
    events = [json.loads(line) for line in (args.student_run / "on-policy-events.jsonl").read_text().splitlines()]
    queries = sorted((row for row in events if row["record_type"] == "query"
                      and row["episode_serial"] in chosen and row["query_in_episode"] in (0, 1, 2)),
                     key=lambda row: (row["episode_serial"], row["query_in_episode"]))
    assert len(queries) == 60
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    manifest = []
    for index, row in enumerate(queries):
        source = args.student_run / row["file"]
        assert file_sha256(source) == row["file_sha256"]
        with np.load(source, allow_pickle=False) as sample:
            image, wrist_image = sample["image"], sample["wrist_image"]
            state = sample["state"].astype(np.float32)
            instruction = str(sample["instruction"].item())
            student_action = sample["student_action"].astype(np.float32)
        assert student_action.shape == (8, 7)
        assert observation_hash(image, wrist_image, state, instruction) == row["observation_sha256"]
        target = args.output / f"sample-{index:05d}.npz"
        np.savez_compressed(target, image=image, wrist_image=wrist_image, state=state,
                            instruction=np.array(instruction), action=student_action[0])
        manifest.append({"file": target.name, "task_id": pairs[row["episode_serial"]]["task_id"],
                         "episode_serial": row["episode_serial"], "query_in_episode": row["query_in_episode"],
                         "observation_sha256": row["observation_sha256"], "source": row["file"],
                         "source_sha256": row["file_sha256"], "sample_sha256": file_sha256(target)})
    (args.output / "manifest.json").write_text(json.dumps({"schema_version": "1.0", "role": "student_visited",
        "selection": "all task_id=1,2 episodes, query0-2; development diagnostic", "count": 60,
        "dataset_action_role": "placeholder; not a teacher target", "samples": manifest}, indent=2) + "\n")


if __name__ == "__main__":
    main()
