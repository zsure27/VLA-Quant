"""Retrospective paired flip audit; never uses outcomes as policy inputs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import tarfile


SHARDS = ("0-9", "10-19", "20-29", "30-39", "40-49")


def read_json(tar: tarfile.TarFile, name: str):
    handle = tar.extractfile(name)
    if handle is None:
        raise ValueError(name)
    return json.load(handle)


def read_jsonl(tar: tarfile.TarFile, name: str):
    handle = tar.extractfile(name)
    if handle is None:
        raise ValueError(name)
    return [json.loads(line) for line in handle]


def mean(values):
    return statistics.mean(values) if values else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--paired-archive", required=True, type=Path)
    p.add_argument("--teacher-archive", required=True, type=Path)
    p.add_argument("--query3-archive", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    episodes = []
    with tarfile.open(args.paired_archive) as paired, tarfile.open(args.teacher_archive) as teacher, tarfile.open(args.query3_archive) as query3:
        paired_names = paired.getnames()
        teacher_names = teacher.getnames()
        for shard in SHARDS:
            pair_name = next(name for name in paired_names if name.startswith(f"eval/p25-paired-headroom-{shard}-") and name.endswith("/paired-episodes.jsonl"))
            teach_name = next(name for name in teacher_names if name.endswith(f"teacher-{shard}/same-observation-comparisons.jsonl"))
            pairs = read_jsonl(paired, pair_name)
            teach_rows = read_jsonl(teacher, teach_name)
            control = f"control-{shard}/"
            metrics = read_json(query3, control + "exact12l-no-lora/metrics.json")
            samples = read_json(query3, control + "samples/manifest.json")["samples"]
            assert len(pairs) == len(metrics) == len(samples) == 100
            by_episode = defaultdict(list)
            for row in teach_rows:
                if row.get("record_type") == "query":
                    by_episode[row["episode_serial"]].append(row)
            assert len(by_episode) == 100
            query3_by_episode = {row["episode_serial"]: row for rows in by_episode.values() for row in rows if row["query_in_episode"] == 3}
            assert len(query3_by_episode) == 100
            for serial, (pair, metric, sample) in enumerate(zip(pairs, metrics, samples)):
                assert sample["episode_serial"] == serial and metric["sample"] == sample["file"]
                rows = sorted(by_episode[serial], key=lambda row: row["query_in_episode"])
                probe = query3_by_episode[serial]
                assert sample["observation_sha256"] == probe["observation_sha256"]
                assert all(row["success"] == pair["C1"] for row in rows)
                if pair["C0"] and not pair["C1"]:
                    category = "new_failure"
                elif not pair["C0"] and pair["C1"]:
                    category = "rescued"
                elif pair["C0"] and pair["C1"]:
                    category = "both_success"
                else:
                    category = "both_fail"
                third = max(1, len(rows) // 3)
                first_gripper = next((row["query_in_episode"] for row in rows if row["gripper_disagreement"] > 0), None)
                episodes.append({
                    "shard": shard, "episode_serial": serial,
                    "task_id": pair["task_id"], "init_state_index": pair["init_state_index"],
                    "category": category, "C0": pair["C0"], "C1": pair["C1"], "C2": pair["C2"],
                    "queries": len(rows),
                    "lora_teacher_mse_episode_mean": mean([row["chunk_mse"] for row in rows]),
                    "lora_teacher_gripper_disagreement_episode_mean": mean([row["gripper_disagreement"] for row in rows]),
                    "late_minus_early_mse": mean([row["chunk_mse"] for row in rows[-third:]]) - mean([row["chunk_mse"] for row in rows[:third]]),
                    "first_gripper_disagreement_query": first_gripper,
                    "query3_lora_mse": probe["chunk_mse"],
                    "query3_no_lora_mse": metric["raw_action"]["mse"],
                    "query3_lora_minus_no_lora_mse": probe["chunk_mse"] - metric["raw_action"]["mse"],
                    "query3_dim_mse_lora_minus_no_lora": [
                        probe["rmse_per_dimension"][index] ** 2 - metric["raw_rmse_per_dim"][index] ** 2
                        for index in range(7)
                    ],
                    "query3_lora_minus_no_lora_gripper_disagreement": probe["gripper_disagreement"] - metric["raw_gripper_disagreement"],
                })
    groups = {}
    for category in ("new_failure", "rescued", "both_success", "both_fail"):
        rows = [row for row in episodes if row["category"] == category]
        groups[category] = {
            "episodes": len(rows),
            "task_counts": dict(sorted(Counter(row["task_id"] for row in rows).items())),
            "mean_queries": mean([row["queries"] for row in rows]),
            "mean_lora_teacher_mse": mean([row["lora_teacher_mse_episode_mean"] for row in rows]),
            "mean_late_minus_early_mse": mean([row["late_minus_early_mse"] for row in rows]),
            "mean_query3_lora_minus_no_lora_mse": mean([row["query3_lora_minus_no_lora_mse"] for row in rows]),
            "query3_lora_better_count": sum(row["query3_lora_minus_no_lora_mse"] < 0 for row in rows),
        }
    output = {
        "classification": "retrospective development mechanism audit; no new rollouts or teacher queries",
        "n": len(episodes), "groups": groups, "episodes": episodes,
        "limitations": [
            "Future outcome labels are used for retrospective grouping only, never model input.",
            "Query3 is not the observed failure onset and cannot establish causality.",
            "Episode lengths and task distributions confound between-group MSE comparisons.",
            "No independent C0/C2 same-state teacher queries or video phase labels in these archives.",
        ],
    }
    assert len(episodes) == 500
    assert [groups[key]["episodes"] for key in ("new_failure", "rescued", "both_success", "both_fail")] == [28, 25, 383, 64]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
