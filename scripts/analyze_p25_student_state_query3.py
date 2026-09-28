"""Leakage-aware same-state query3 gate for student-state-distilled Recovery-LoRA."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import tarfile


SHARDS = ("0-9", "10-19", "20-29", "30-39", "40-49")


def read(tar, name):
    stream = tar.extractfile(name)
    if stream is None:
        raise ValueError(name)
    return json.load(stream)


def average(values):
    return statistics.mean(values) if values else None


def summarize(rows):
    return {"n": len(rows), "mean_mse": {name: average([r[name + "_mse"] for r in rows])
             for name in ("new", "old", "exact12l", "static14l")},
            "new_better_than_old": sum(r["new_mse"] < r["old_mse"] for r in rows),
            "mean_gripper_disagreement": {name: average([r[name + "_gripper"] for r in rows])
             for name in ("new", "old", "exact12l", "static14l")},
            "mean_dim_mse": {name: [average([r[name + "_dim_mse"][index] for r in rows]) for index in range(7)]
             for name in ("new", "old", "exact12l", "static14l")}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("new", "control", "teacher", "static", "paired"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    rows = []
    with tarfile.open(args.new) as new, tarfile.open(args.control) as control, tarfile.open(args.teacher) as teacher, tarfile.open(args.static) as static, tarfile.open(args.paired) as paired:
        pair_names, teacher_names = paired.getnames(), teacher.getnames()
        for shard in SHARDS:
            prefix = f"control-{shard}/"
            samples = read(control, prefix + "samples/manifest.json")["samples"]
            baseline = read(control, prefix + "exact12l-no-lora/metrics.json")
            reference = read(static, f"reference-{shard}/static14l/metrics.json")
            candidate = read(new, f"query3-student-state-{shard}/candidate/metrics.json")
            teacher_name = next(n for n in teacher_names if n.endswith(f"teacher-{shard}/same-observation-comparisons.jsonl"))
            teacher_rows = {r["episode_serial"]: r for r in
                            (json.loads(line) for line in teacher.extractfile(teacher_name) if line)
                            if r.get("record_type") == "query" and r["query_in_episode"] == 3}
            pair_name = next(n for n in pair_names if n.startswith(f"eval/p25-paired-headroom-{shard}-") and n.endswith("/paired-episodes.jsonl"))
            pairs = [json.loads(line) for line in paired.extractfile(pair_name) if line]
            assert len(samples) == len(baseline) == len(reference) == len(candidate) == len(pairs) == 100
            for serial, (sample, b, s, c, pair) in enumerate(zip(samples, baseline, reference, candidate, pairs)):
                assert sample["episode_serial"] == serial
                assert sample["file"] == b["sample"] == s["sample"] == c["sample"]
                old = teacher_rows[serial]
                assert old["observation_sha256"] == sample["observation_sha256"]
                reset = pair["init_state_index"]
                if reset < 5:
                    assert shard == "0-9"
                    continue
                category = ("new_failure" if pair["C0"] and not pair["C1"] else
                            "rescued" if not pair["C0"] and pair["C1"] else
                            "both_success" if pair["C0"] else "both_fail")
                rows.append({"shard": shard, "episode_serial": serial, "task_id": pair["task_id"],
                    "init_state_index": reset, "category": category,
                    "old_mse": old["chunk_mse"], "new_mse": c["raw_action"]["mse"],
                    "exact12l_mse": b["raw_action"]["mse"], "static14l_mse": s["raw_action"]["mse"],
                    "old_gripper": old["gripper_disagreement"], "new_gripper": c["raw_gripper_disagreement"],
                    "exact12l_gripper": b["raw_gripper_disagreement"], "static14l_gripper": s["raw_gripper_disagreement"],
                    "old_dim_mse": [x*x for x in old["rmse_per_dimension"]],
                    "new_dim_mse": [x*x for x in c["raw_rmse_per_dim"]],
                    "exact12l_dim_mse": [x*x for x in b["raw_rmse_per_dim"]],
                    "static14l_dim_mse": [x*x for x in s["raw_rmse_per_dim"]]})
    assert len(rows) == 450 and {r["init_state_index"] for r in rows} == set(range(5, 50))
    rng = random.Random(20260929)
    deltas = [r["new_mse"] - r["old_mse"] for r in rows]
    boots = sorted(average(rng.choices(deltas, k=450)) for _ in range(3000))
    output = {"schema_version": "1.0", "classification": "P2.5 development same-student-state diagnostic; no new closed loop",
              "evaluation_resets": "5-49", "excluded_train_and_adjacent_resets": "0-4",
              "overall": summarize(rows), "mean_new_minus_old_mse": average(deltas),
              "paired_episode_bootstrap_95ci": [boots[74], boots[2924]],
              "by_task": {str(task): summarize([r for r in rows if r["task_id"] == task]) for task in range(10)},
              "posthoc_by_outcome": {cat: summarize([r for r in rows if r["category"] == cat])
                for cat in ("new_failure", "rescued", "both_success", "both_fail")},
              "source_sha256": {key: hashlib.sha256(getattr(args,key).read_bytes()).hexdigest()
                for key in ("new", "control", "teacher", "static", "paired")},
              "limitations": ["All states are induced by old LoRA, not the new candidate.",
                "Query3 is not necessarily failure onset or an unbiased closed-loop endpoint.",
                "Resets0-4 were excluded because student-state training used resets0-3.",
                "Retrospective outcome categories are explanatory only and never policy inputs."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"overall": output["overall"], "ci": output["paired_episode_bootstrap_95ci"]}))


if __name__ == "__main__":
    main()
