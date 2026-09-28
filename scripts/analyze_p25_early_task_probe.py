"""Analyze preregistered early-query controls, paired at student observations."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import tarfile


SHARDS = ("0-9", "10-19", "20-29", "30-39", "40-49")


def read(tar, name):
    member = tar.extractfile(name)
    if member is None:
        raise ValueError(name)
    return json.load(member)


def mean(values):
    return statistics.mean(values) if values else None


def summarize(rows):
    return {"n": len(rows), "episodes": len({(row["shard"], row["episode_serial"]) for row in rows}),
            "mean_mse": {name: mean([row[name + "_mse"] for row in rows]) for name in ("lora", "exact12l", "static14l")},
            "lora_better_than_exact12l": sum(row["lora_mse"] < row["exact12l_mse"] for row in rows),
            "lora_better_than_static14l": sum(row["lora_mse"] < row["static14l_mse"] for row in rows),
            "mean_gripper_disagreement": {name: mean([row[name + "_gripper_disagreement"] for row in rows])
                for name in ("lora", "exact12l", "static14l")},
            "mean_dim_mse": {name: [mean([row[name + "_dim_mse"][i] for row in rows]) for i in range(7)]
                for name in ("lora", "exact12l", "static14l")}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--early-archive", required=True, type=Path)
    p.add_argument("--teacher-archive", required=True, type=Path)
    p.add_argument("--paired-archive", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    rows = []
    with tarfile.open(args.early_archive) as early, tarfile.open(args.teacher_archive) as teacher, tarfile.open(args.paired_archive) as paired:
        teacher_names = teacher.getnames()
        pair_names = paired.getnames()
        for shard in SHARDS:
            prefix = f"probe-{shard}/"
            samples = read(early, prefix + "samples/manifest.json")["samples"]
            exact = read(early, prefix + "exact12l-no-lora/metrics.json")
            static = read(early, prefix + "static14l/metrics.json")
            teacher_name = next(name for name in teacher_names if name.endswith(f"teacher-{shard}/same-observation-comparisons.jsonl"))
            teach_rows = [json.loads(line) for line in teacher.extractfile(teacher_name) if line]
            teach = {(row["episode_serial"], row["query_in_episode"]): row for row in teach_rows if row.get("record_type") == "query"}
            pair_name = next(name for name in pair_names if name.startswith(f"eval/p25-paired-headroom-{shard}-") and name.endswith("/paired-episodes.jsonl"))
            pair_rows = [json.loads(line) for line in paired.extractfile(pair_name) if line]
            assert len(samples) == len(exact) == len(static) == 60 and len(pair_rows) == 100
            for sample, a, b in zip(samples, exact, static):
                assert sample["file"] == a["sample"] == b["sample"]
                serial, query = sample["episode_serial"], sample["query_in_episode"]
                original = teach[(serial, query)]
                assert original["observation_sha256"] == sample["observation_sha256"]
                assert pair_rows[serial]["task_id"] == sample["task_id"]
                rows.append({"shard": shard, "episode_serial": serial, "task_id": sample["task_id"],
                    "query_in_episode": query, "C0": pair_rows[serial]["C0"], "C1": pair_rows[serial]["C1"],
                    "lora_mse": original["chunk_mse"], "exact12l_mse": a["raw_action"]["mse"],
                    "static14l_mse": b["raw_action"]["mse"],
                    "lora_gripper_disagreement": original["gripper_disagreement"],
                    "exact12l_gripper_disagreement": a["raw_gripper_disagreement"],
                    "static14l_gripper_disagreement": b["raw_gripper_disagreement"],
                    "lora_dim_mse": [x*x for x in original["rmse_per_dimension"]],
                    "exact12l_dim_mse": [x*x for x in a["raw_rmse_per_dim"]],
                    "static14l_dim_mse": [x*x for x in b["raw_rmse_per_dim"]]})
    assert len(rows) == 300
    output = {"schema_version": "1.0", "classification": "P2.5 observational same-LoRA-state diagnosis",
              "selection": "all episodes in development task1 and task2, query0-2",
              "overall": summarize(rows),
              "by_task_query": {str(task): {str(query): summarize([r for r in rows if r["task_id"] == task and r["query_in_episode"] == query])
                    for query in range(3)} for task in (1, 2)},
              "by_task": {str(task): summarize([r for r in rows if r["task_id"] == task]) for task in (1, 2)},
              "posthoc_by_task_outcome": {str(task): {category: summarize([r for r in rows if r["task_id"] == task and
                    ((r["C0"] and not r["C1"]) if category == "new_failure" else (not r["C0"] and r["C1"]))])
                    for category in ("new_failure", "rescued")} for task in (1, 2)},
              "source_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in
                    (("early", args.early_archive), ("teacher", args.teacher_archive), ("paired", args.paired_archive))},
              "limitations": ["Same states were visited by LoRA, not independent rollouts.",
                              "Task selection follows historical task-level findings and is diagnostic, not an unbiased gate.",
                              "Closed-loop success and BF16 MSE are different endpoints.",
                              "Outcome categories are post hoc and excluded from probe selection and policy inputs."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"overall": output["overall"], "by_task_query": output["by_task_query"]}))


if __name__ == "__main__":
    main()
