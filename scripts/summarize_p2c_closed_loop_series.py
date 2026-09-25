"""Combine disjoint P2C paired-summary shards without re-reading rollout logs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_p2c_closed_loop import bootstrap, paired_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    identities = set()
    sources = []
    for directory in args.directories:
        path = directory / "paired-episodes.jsonl"
        shard = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for row in shard:
            identity = (row["task_id"], row["init_state_index"])
            if identity in identities:
                raise ValueError(f"duplicate paired identity: {identity}")
            identities.add(identity)
            rows.append(row)
        sources.append({"directory": str(directory), "episodes": len(shard)})
    rows.sort(key=lambda row: (row["task_id"], row["init_state_index"]))
    rates = {case: sum(row[case] for row in rows) / len(rows) for case in ("C0", "C1", "C2")}
    denominator = rates["C2"] - rates["C0"]
    summary = {
        "schema_version": "1.0",
        "classification": "historical/development paired closed-loop evidence",
        "episodes_per_config": len(rows),
        "successes": {case: sum(row[case] for row in rows) for case in ("C0", "C1", "C2")},
        "success_rates": rates,
        "recovery_fraction": (rates["C1"] - rates["C0"]) / denominator if denominator > 0 else None,
        "recovery_fraction_denominator_positive": denominator > 0,
        "paired_C0_C1": paired_matrix(rows, "C0", "C1"),
        "paired_C1_C2": paired_matrix(rows, "C1", "C2"),
        "bootstrap": bootstrap(rows),
        "sources": sources,
    }
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
