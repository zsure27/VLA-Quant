"""Freeze 80 student-visited training observations from reset indices 0-3."""
from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path

import numpy as np

from qvla.on_policy_capture import file_sha256, observation_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paired-shard", required=True, type=Path)
    p.add_argument("--student-run", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    pairs = [json.loads(line) for line in (args.paired_shard / "paired-episodes.jsonl").read_text().splitlines()]
    assert len(pairs) == 100 and {r["init_state_index"] for r in pairs} == set(range(10))
    selected_episodes = {serial for serial, pair in enumerate(pairs) if pair["init_state_index"] in (0, 1, 2, 3)}
    assert len(selected_episodes) == 40
    events = [json.loads(line) for line in (args.student_run / "on-policy-events.jsonl").read_text().splitlines()]
    queries = sorted((row for row in events if row["record_type"] == "query" and
                      row["episode_serial"] in selected_episodes and row["query_in_episode"] in (0, 1)),
                     key=lambda row: (row["episode_serial"], row["query_in_episode"]))
    assert len(queries) == 80
    assert Counter(pairs[row["episode_serial"]]["task_id"] for row in queries) == {task: 8 for task in range(10)}
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
            action = sample["student_action"].astype(np.float32)
        assert action.shape == (8, 7)
        assert observation_hash(image, wrist_image, state, instruction) == row["observation_sha256"]
        target = args.output / f"sample-{index:05d}.npz"
        np.savez_compressed(target, image=image, wrist_image=wrist_image, state=state,
                            instruction=np.array(instruction), action=action[0])
        manifest.append({"file": target.name, "task_id": pairs[row["episode_serial"]]["task_id"],
                         "init_state_index": pairs[row["episode_serial"]]["init_state_index"],
                         "episode_serial": row["episode_serial"], "query_in_episode": row["query_in_episode"],
                         "observation_sha256": row["observation_sha256"], "source": row["file"],
                         "source_sha256": row["file_sha256"], "sample_sha256": file_sha256(target)})
    (args.output / "manifest.json").write_text(json.dumps({"schema_version": "1.0", "role": "student_state_train",
        "selection": "all tasks; reset indices 0-3; query0-1; 8 samples per task",
        "count": 80, "excluded_evaluation_resets": "0-4; evaluate only 5-49",
        "dataset_action_role": "placeholder only; frozen BF16 is queried on same observation for training target",
        "samples": manifest}, indent=2) + "\n")


if __name__ == "__main__":
    main()
