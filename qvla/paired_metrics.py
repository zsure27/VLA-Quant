"""Paired episode metrics with dynamic task-cluster bootstrap denominator."""
from __future__ import annotations
from collections import defaultdict
import random


def comparison(rows: list[dict], candidate: str, baseline: str, draws: int = 20000) -> dict:
    if not rows:
        raise ValueError("Empty paired cohort")
    keys = [(r["task_id"], r["init_state_index"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate task/reset; do not pool repeated cohorts")
    clusters = defaultdict(list)
    for row in rows:
        if type(row[candidate]) is not bool or type(row[baseline]) is not bool:
            raise ValueError("Success labels must be booleans")
        clusters[row["task_id"]].append(int(row[candidate]) - int(row[baseline]))
    tasks = sorted(clusters)
    delta = {str(t): sum(clusters[t]) for t in tasks}
    rng = random.Random(20260930)
    values = []
    for _ in range(draws):
        sample = [rng.choice(tasks) for _ in tasks]
        values.append(sum(sum(clusters[t]) for t in sample) / sum(len(clusters[t]) for t in sample))
    values.sort()
    rescue = sum(not r[baseline] and r[candidate] for r in rows)
    broken = sum(r[baseline] and not r[candidate] for r in rows)
    n = rescue + broken
    from math import comb
    p = min(1.0, 2 * sum(comb(n, k) for k in range(min(rescue, broken) + 1)) / 2**n) if n else 1.0
    return {"episodes": len(rows), "rescue": rescue, "break": broken, "net": rescue-broken,
            "delta_fraction": (rescue-broken)/len(rows), "exact_mcnemar_p": p,
            "task_cluster_bootstrap_95ci_fraction": [values[int(.025*draws)], values[int(.975*draws)]],
            "bootstrap_seed": 20260930, "bootstrap_draws": draws, "delta_by_task": delta,
            "tasks_improved": sum(d > 0 for d in delta.values()),
            "tasks_harmed": sum(d < 0 for d in delta.values())}
