"""Pair 14L same-observation diagnostic with prior LoRA and exact-12L controls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics
import tarfile


SHARDS = ("0-9", "10-19", "20-29", "30-39", "40-49")


def read(tar, name):
    member = tar.extractfile(name)
    if member is None:
        raise ValueError(name)
    return json.load(member)


def avg(rows, key):
    return statistics.mean(row[key] for row in rows) if rows else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--control", required=True, type=Path)
    p.add_argument("--flip-audit", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    flips = json.loads(args.flip_audit.read_text())["episodes"]
    lookup = {(row["shard"], row["episode_serial"]): row for row in flips}
    assert len(lookup) == 500
    rows = []
    with tarfile.open(args.reference) as ref, tarfile.open(args.control) as control:
        for shard in SHARDS:
            static = read(ref, f"reference-{shard}/static14l/metrics.json")
            no_lora = read(control, f"control-{shard}/exact12l-no-lora/metrics.json")
            samples = read(control, f"control-{shard}/samples/manifest.json")["samples"]
            assert len(static) == len(no_lora) == len(samples) == 100
            for serial, (s, n, sample) in enumerate(zip(static, no_lora, samples)):
                assert s["sample"] == n["sample"] == sample["file"]
                flip = lookup[(shard, serial)]
                rows.append({"shard": shard, "episode_serial": serial, "task_id": flip["task_id"],
                             "category": flip["category"], "lora_mse": flip["query3_lora_mse"],
                             "exact12l_mse": n["raw_action"]["mse"],
                             "static14l_mse": s["raw_action"]["mse"],
                             "lora_gripper_disagreement": n["raw_gripper_disagreement"] + flip["query3_lora_minus_no_lora_gripper_disagreement"],
                             "exact12l_gripper_disagreement": n["raw_gripper_disagreement"],
                             "static14l_gripper_disagreement": s["raw_gripper_disagreement"]})
    def summarize(items):
        return {"n": len(items), "mean_mse": {key: avg(items, key + "_mse") for key in ("lora", "exact12l", "static14l")},
                "mean_gripper_disagreement": {key: avg(items, key + "_gripper_disagreement")
                    for key in ("lora", "exact12l", "static14l")},
                "lora_closer_than_static14l": sum(row["lora_mse"] < row["static14l_mse"] for row in items),
                "static14l_closer_than_exact12l": sum(row["static14l_mse"] < row["exact12l_mse"] for row in items)}
    rng = random.Random(20260929)
    deltas = [row["lora_mse"] - row["static14l_mse"] for row in rows]
    boots = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(3000))
    output = {"schema_version": "1.0", "classification": "P2.5 same-LoRA-state observational diagnostic",
              "selection": "all 500 LoRA rollouts, query_in_episode=3", "overall": summarize(rows),
              "mean_delta_lora_minus_static14l_mse": statistics.mean(deltas),
              "episode_bootstrap_95ci": [boots[74], boots[2924]],
              "by_task": {str(task): summarize([row for row in rows if row["task_id"] == task]) for task in range(10)},
              "by_outcome_category": {category: summarize([row for row in rows if row["category"] == category])
                    for category in ("new_failure", "rescued", "both_success", "both_fail")},
              "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
                    (("reference", args.reference), ("control", args.control), ("flip_audit", args.flip_audit))},
              "limitations": ["Same observations are induced by LoRA; probe outputs are not counterfactual closed-loop outcomes.",
                  "Query3 is a fixed temporal sample, not necessarily the first cause of failure.",
                  "Outcome categories are retrospective and must not become policy inputs or a model-selection gate."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key not in ("by_task", "by_outcome_category")}))


if __name__ == "__main__":
    main()
