"""Aggregate frozen-BF16 same-observation comparisons on 500 LoRA rollouts."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics as stats
import tarfile


def mean(values):
    return sum(values) / len(values) if values else None


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows, sources = [], []
    with tarfile.open(args.archive, "r:gz") as archive:
        for member in archive.getmembers():
            if member.isfile() and member.name.endswith("/same-observation-comparisons.jsonl"):
                stream = archive.extractfile(member)
                assert stream is not None
                part = [json.loads(line) for line in stream if line.strip()]
                sources.append({"file": member.name, "queries": len(part)})
                shard = member.name.split("/")[-2]
                rows.extend({"shard": shard, **row} for row in part)
    groups = defaultdict(list)
    for row in rows:
        if row["record_type"] != "query" or row["chunk_execution_steps"] != 8:
            raise RuntimeError("Unexpected record or chunk execution contract")
        groups[(row["shard"], row["episode_serial"])].append(row)
    if len(sources) != 5 or len(groups) != 500:
        raise RuntimeError(f"Expected 5 shards and 500 episodes: {len(sources)}, {len(groups)}")
    episodes = []
    for (shard, serial), group in sorted(groups.items()):
        group.sort(key=lambda row: row["query_in_episode"])
        if [row["query_in_episode"] for row in group] != list(range(len(group))):
            raise RuntimeError(f"Noncontiguous queries: {shard}/{serial}")
        if len({row["success"] for row in group}) != 1:
            raise RuntimeError("Outcome drift inside episode")
        third = max(1, len(group) // 3)
        episodes.append({
            "shard": shard, "serial": serial, "task": group[0]["task"],
            "success": bool(group[0]["success"]), "queries": len(group),
            "mean_chunk_mse": mean([row["chunk_mse"] for row in group]),
            "mean_gripper_disagreement": mean([row["gripper_disagreement"] for row in group]),
            "late_minus_early_mse": mean([row["chunk_mse"] for row in group[-third:]]) -
                                    mean([row["chunk_mse"] for row in group[:third]]),
            "first_gripper_disagreement_query": next(
                (row["query_in_episode"] for row in group if row["gripper_disagreement"] > 0), None),
        })
    success = [e for e in episodes if e["success"]]
    failure = [e for e in episodes if not e["success"]]
    if len(success) != 408 or len(failure) != 92:
        raise RuntimeError("Student outcomes disagree with paired Spatial500")
    rng = random.Random(20260928)
    contrasts = []
    for _ in range(10000):
        left = mean([failure[rng.randrange(len(failure))]["mean_chunk_mse"] for _ in failure])
        right = mean([success[rng.randrange(len(success))]["mean_chunk_mse"] for _ in success])
        contrasts.append(left - right)
    contrasts.sort()
    task_groups = defaultdict(list)
    for episode in episodes:
        task_groups[episode["task"]].append(episode)
    result = {
        "schema_version": "1.0", "classification": "P2.5 observational mechanism diagnosis; development resets",
        "archive_sha256": digest(args.archive), "sources": sorted(sources, key=lambda row: row["file"]),
        "queries": len(rows), "episodes": len(episodes), "successes": len(success),
        "query_mean_chunk_mse": mean([r["chunk_mse"] for r in rows]),
        "query_median_chunk_mse": stats.median(r["chunk_mse"] for r in rows),
        "query_mean_gripper_disagreement": mean([r["gripper_disagreement"] for r in rows]),
        "episode_mean_chunk_mse": mean([e["mean_chunk_mse"] for e in episodes]),
        "success_episode_mean_mse": mean([e["mean_chunk_mse"] for e in success]),
        "failure_episode_mean_mse": mean([e["mean_chunk_mse"] for e in failure]),
        "failure_minus_success_episode_mse_95ci": [contrasts[250], contrasts[9750]],
        "worst_decile_episode_mse_threshold": sorted(e["mean_chunk_mse"] for e in episodes)[450],
        "late_minus_early_episode_mse_mean": mean([e["late_minus_early_mse"] for e in episodes]),
        "late_worse_episodes": sum(e["late_minus_early_mse"] > 0 for e in episodes),
        "first_gripper_disagreement_episode_fraction": mean([
            e["first_gripper_disagreement_query"] is not None for e in episodes]),
        "per_chunk_step_mean_rmse": [mean([r["rmse_per_step"][i] for r in rows]) for i in range(8)],
        "per_action_dimension_mean_rmse": [mean([r["rmse_per_dimension"][i] for r in rows]) for i in range(7)],
        "student_gripper_margin_abs_lt_0_1": mean([
            abs(x) < 0.1 for r in rows for x in r["student_gripper_signed_margin"]]),
        "teacher_gripper_margin_abs_lt_0_1": mean([
            abs(x) < 0.1 for r in rows for x in r["teacher_gripper_signed_margin"]]),
        "per_task": [{"task": task, "episodes": len(group),
                      "successes": sum(e["success"] for e in group),
                      "episode_mean_mse": mean([e["mean_chunk_mse"] for e in group]),
                      "episode_gripper_disagreement": mean([e["mean_gripper_disagreement"] for e in group])}
                     for task, group in sorted(task_groups.items())],
        "limitations": ["No independent failure-onset annotation; gripper disagreement is not failure causation.",
                        "No preregistered large-divergence threshold; first large divergence remains null.",
                        "Student-visited query frequency differs from trajectory-level offline sampling."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("queries", "episodes", "query_mean_chunk_mse",
                                               "query_mean_gripper_disagreement", "failure_minus_success_episode_mse_95ci")}))


if __name__ == "__main__":
    main()
