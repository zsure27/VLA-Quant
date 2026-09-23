"""Audit a finite P0 rollout without changing its original exit receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    consoles = list(args.run.glob("*/console.log"))
    traces = list(args.run.glob("*/policy-queries.jsonl"))
    eval_logs = list(args.run.glob("*/EVAL-*.txt"))
    if len(consoles) != 1 or len(traces) != 1 or len(eval_logs) != 1:
        raise RuntimeError("expected exactly one console, trace, and EVAL log")
    console = consoles[0].read_text(encoding="utf-8", errors="replace")
    totals = re.findall(r"Total episodes:\s+(\d+)", console)
    successes = re.findall(r"Total successes:\s+(\d+)", console)
    if not totals or not successes:
        raise RuntimeError("missing final evaluator totals")
    records = []
    with traces[0].open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            chunk, state = record["raw_policy_chunk"], record["state"]
            if record.get("finite") is not True or len(chunk) != 8 or any(len(row) != 7 for row in chunk):
                raise RuntimeError("invalid action trace shape or finite flag")
            if len(state) != 8 or not all(math.isfinite(float(value)) for row in chunk for value in row):
                raise RuntimeError("non-finite action or invalid state shape")
            if not all(math.isfinite(float(value)) for value in state):
                raise RuntimeError("non-finite proprio state")
            records.append(record)
    receipt = (args.run / "exit-code.txt").read_text().strip() if (args.run / "exit-code.txt").exists() else None
    result = {
        "evaluator_final_episodes": int(totals[-1]),
        "evaluator_final_successes": int(successes[-1]),
        "policy_queries": len(records),
        "all_policy_actions_and_states_finite": True,
        "wrapper_exit_code": int(receipt) if receipt is not None else None,
        "console_sha256": hashlib.sha256(consoles[0].read_bytes()).hexdigest(),
        "trace_sha256": hashlib.sha256(traces[0].read_bytes()).hexdigest(),
        "eval_log_sha256": hashlib.sha256(eval_logs[0].read_bytes()).hexdigest(),
    }
    if result["evaluator_final_episodes"] != 50 or not records:
        raise RuntimeError(f"incomplete P0 rollout: {result}")
    output = args.output or args.run / "posthoc-rollout-audit.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
