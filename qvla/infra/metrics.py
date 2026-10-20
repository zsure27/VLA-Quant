"""Strict episode identities and descriptive, suite-aware aggregation."""
from __future__ import annotations

from collections import defaultdict
import csv
import math
from pathlib import Path
from .common import read_json, write_json


def summarize(rows):
    seen, groups, costs = set(), defaultdict(list), defaultdict(lambda: {"queries": 0, "elapsed_seconds": 0.0})
    excluded = []
    for row in rows:
        key = tuple(row[k] for k in ("cohort_id", "suite", "task_id", "reset", "model_id"))
        if key in seen:
            raise ValueError("Duplicate episode across rounds: " + repr(key))
        seen.add(key)
        # Interrupted/error records survive; neither becomes a completed failure label.
        if row.get("status") != "completed":
            excluded.append(row)
            continue
        if type(row.get("success")) is not bool or row.get("finite") is not True:
            raise ValueError("Completed episodes need boolean success and finite=true")
        for name in ("queries", "elapsed_seconds"):
            value = row.get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid cost")
        if type(row["queries"]) is not int:
            raise ValueError("queries must be an explicit integer")
        group = (row["cohort_id"], row["suite"], row["model_id"])
        groups[group].append(row)
        for name in ("queries", "elapsed_seconds"):
            costs[group][name] += row.get(name, 0)
    scores = [{"cohort_id": k[0], "suite": k[1], "model_id": k[2], "episodes": len(v),
               "successes": sum(r["success"] for r in v),
               "success_rate": sum(r["success"] for r in v) / len(v), **costs[k],
               "failures": [{"task_id": r["task_id"], "reset": r["reset"]} for r in v if not r["success"]]}
              for k, v in sorted(groups.items())]
    return {"schema_version": 1, "scores": scores, "excluded": excluded,
            "note": "Exploratory evidence; no cross-cohort causal subtraction or automatic holdout claims"}


def paired(rows, baseline, candidate):
    summarize(rows)  # reject duplicate/conflicting or malformed observations first
    left, right = {}, {}
    for row in rows:
        if row.get("status") != "completed":
            continue
        key = tuple(row[k] for k in ("cohort_id", "suite", "task_id", "reset", "env_seed", "policy_seed", "protocol_sha256"))
        if row["model_id"] == baseline:
            left[key] = row
        elif row["model_id"] == candidate:
            right[key] = row
    if not left or set(left) != set(right):
        raise ValueError("Require full matching cohort/suite/task/reset/seeds/protocol; report missing pairs separately")
    buckets = defaultdict(list)
    for key in left:
        buckets[key[:2]].append({"task_id": key[2], "init_state_index": key[3],
                                "baseline": left[key]["success"], "candidate": right[key]["success"]})
    from qvla.paired_metrics import comparison
    return [{"cohort_id": key[0], "suite": key[1], **comparison(value, "candidate", "baseline")}
            for key, value in sorted(buckets.items())]


def export(input_path, output_path):
    rows = read_json(input_path)
    result = summarize(rows)
    write_json(output_path, result)
    csv_path = Path(output_path).with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        keys = ["cohort_id", "suite", "model_id", "episodes", "successes", "success_rate", "queries", "elapsed_seconds"]
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["scores"])
    return result
