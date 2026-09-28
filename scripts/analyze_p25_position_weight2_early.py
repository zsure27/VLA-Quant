"""Pair old and position-weighted LoRA on identical early student states."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tarfile


SHARDS = ("0-9", "10-19", "20-29", "30-39", "40-49")


def read(tar, name):
    stream = tar.extractfile(name)
    if stream is None:
        raise ValueError(name)
    return json.load(stream)


def summary(rows):
    return {"n": len(rows), "episodes": len({(r["shard"], r["episode_serial"]) for r in rows}),
            "mean_mse": {key: statistics.mean(r[key + "_mse"] for r in rows) for key in ("old", "pos2", "exact12l")},
            "pos2_better_than_old": sum(r["pos2_mse"] < r["old_mse"] for r in rows),
            "mean_gripper_disagreement": {key: statistics.mean(r[key + "_gripper"] for r in rows)
                for key in ("old", "pos2", "exact12l")},
            "mean_dim_mse": {key: [statistics.mean(r[key + "_dim_mse"][index] for r in rows) for index in range(7)]
                for key in ("old", "pos2", "exact12l")}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--early-archive", required=True, type=Path)
    p.add_argument("--pos2-archive", required=True, type=Path)
    p.add_argument("--teacher-archive", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    rows = []
    with tarfile.open(args.early_archive) as early, tarfile.open(args.pos2_archive) as new, tarfile.open(args.teacher_archive) as teacher:
        teacher_names = teacher.getnames()
        for shard in SHARDS:
            prefix = f"probe-{shard}/"
            samples = read(early, prefix + "samples/manifest.json")["samples"]
            base = read(early, prefix + "exact12l-no-lora/metrics.json")
            candidate = read(new, f"early-task-pos2-{shard}/candidate/metrics.json")
            teacher_name = next(name for name in teacher_names if name.endswith(f"teacher-{shard}/same-observation-comparisons.jsonl"))
            original = {(r["episode_serial"], r["query_in_episode"]): r for r in
                        (json.loads(line) for line in teacher.extractfile(teacher_name) if line)
                        if r.get("record_type") == "query"}
            assert len(samples) == len(base) == len(candidate) == 60
            for sample, b, c in zip(samples, base, candidate):
                assert sample["file"] == b["sample"] == c["sample"]
                row = original[(sample["episode_serial"], sample["query_in_episode"])]
                assert sample["observation_sha256"] == row["observation_sha256"]
                rows.append({"shard": shard, "episode_serial": sample["episode_serial"],
                             "task_id": sample["task_id"], "query": sample["query_in_episode"],
                             "old_mse": row["chunk_mse"], "pos2_mse": c["raw_action"]["mse"],
                             "exact12l_mse": b["raw_action"]["mse"],
                             "old_gripper": row["gripper_disagreement"],
                             "pos2_gripper": c["raw_gripper_disagreement"],
                             "exact12l_gripper": b["raw_gripper_disagreement"],
                             "old_dim_mse": [x*x for x in row["rmse_per_dimension"]],
                             "pos2_dim_mse": [x*x for x in c["raw_rmse_per_dim"]],
                             "exact12l_dim_mse": [x*x for x in b["raw_rmse_per_dim"]]})
    assert len(rows) == 300
    output = {"schema_version": "1.0", "classification": "P2.5 same-LoRA-state offline candidate gate",
              "overall": summary(rows),
              "by_task_query": {str(task): {str(query): summary([r for r in rows if r["task_id"] == task and r["query"] == query])
                   for query in range(3)} for task in (1, 2)},
              "by_task": {str(task): summary([r for r in rows if r["task_id"] == task]) for task in (1, 2)},
              "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
                   (("early", args.early_archive), ("pos2", args.pos2_archive), ("teacher", args.teacher_archive))},
              "limitations": ["Student states were induced by the old LoRA, so no counterfactual closed-loop inference follows.",
                  "Only two tasks were chosen for mechanism diagnosis; independent paired rollout is still required."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"overall": output["overall"], "by_task_query": output["by_task_query"]}))


if __name__ == "__main__":
    main()
