"""Summarize the preregistered 450 paired P2.5 rollouts, never merge train resets 0-4."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random


CASES = ("C0", "C1", "C2", "C3")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired(a, b, rows):
    gain = sum(not x[a] and x[b] for x in rows)
    loss = sum(x[a] and not x[b] for x in rows)
    n = gain + loss
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(gain, loss) + 1)) / 2**n) if n else 1.0
    return {"gain": gain, "loss": loss, "net": gain - loss, "discordant": n, "mcnemar_exact_two_sided_p": p}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs=5, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows, sources = [], []
    for shard in args.shards:
        summary = json.loads((shard / "paired-summary.json").read_text())
        file = shard / "paired-episodes.jsonl"
        part = [json.loads(line) for line in file.read_text().splitlines()]
        assert len(part) == summary["episodes"] == summary["count"] * 10
        assert summary["holdout_touched"] is False
        assert all(sum(int(x[c]) for x in part) == summary["successes"][c] for c in CASES)
        rows.extend(part)
        sources.append({"directory": shard.name, "rows_sha256": sha(file), "summary_sha256": sha(shard / "paired-summary.json"),
                        "candidate_eval_sha256": summary["candidate_eval_sha256"]})
    keys = [(x["task_id"], x["init_state_index"]) for x in rows]
    assert len(rows) == len(set(keys)) == 450
    assert set(keys) == {(t, i) for t in range(10) for i in range(5, 50)}
    assert all(not x.get("C3_episode_errors") for x in rows)
    counts = {c: sum(int(x[c]) for x in rows) for c in CASES}
    by_task = {}
    for task in range(10):
        part = [x for x in rows if x["task_id"] == task]
        by_task[str(task)] = {"n": len(part), "successes": {c: sum(int(x[c]) for x in part) for c in CASES},
                              "C3_vs_C0": paired("C0", "C3", part), "C3_vs_C1": paired("C1", "C3", part)}
    rng = random.Random(20260929)
    boot = []
    for _ in range(10000):
        sample = [rows[rng.randrange(len(rows))] for __ in rows]
        boot.append(sum(int(x["C3"]) - int(x["C0"]) for x in sample) / len(sample))
    boot.sort()
    headroom = counts["C2"] - counts["C0"]
    out = {"schema_version": "1.0", "classification": "development paired; reset5-49 only; offline_final_holdout untouched",
           "n": len(rows), "successes": counts, "success_rates": {c: counts[c] / len(rows) for c in CASES},
           "C3_vs_C0": paired("C0", "C3", rows), "C3_vs_C1": paired("C1", "C3", rows),
           "C3_vs_C2": paired("C2", "C3", rows), "C2_vs_C0": paired("C0", "C2", rows),
           "C3_minus_C0_rate": (counts["C3"] - counts["C0"]) / len(rows),
           "C3_minus_C0_bootstrap_95": [boot[250], boot[9750]],
           "static_headroom_C2_minus_C0": headroom,
           "recovery_fraction": (counts["C3"] - counts["C0"]) / headroom if headroom > 0 else None,
           "per_task": by_task, "sources": sources}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("n", "successes", "C3_vs_C0", "C3_vs_C1", "C3_vs_C2", "C3_minus_C0_bootstrap_95", "recovery_fraction")}, indent=2))


if __name__ == "__main__":
    main()
