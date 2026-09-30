"""Audit one 059 five-policy phase before continuing the registered evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import subprocess
from pathlib import Path

CASES = ("C0", "C3", "LW", "BF16", "W4")
MANIFEST = re.compile(r"EPISODE_MANIFEST (\{.*\})")
SUCCESS = re.compile(r"Success: (True|False)")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def flag(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("micro", "first50", "first50trace"), default="micro")
    args = parser.parse_args()
    base = args.root / f"eval-{args.phase}"
    offset, per_task = (20, 1) if args.phase == "micro" else (20, 5)
    expected = per_task * 10
    train = args.root / "language-w2-all-r8-e2e1000"
    assert train.joinpath("exit-code.txt").read_text().strip() == "0"
    assert train.joinpath("e2e_adapter_state.pt").is_file()
    assert train.joinpath("steps-preregistered.txt").read_text().strip() == "1000"

    output = {"gate": f"PASS_PROTOCOL_{args.phase.upper()}", "pairs": expected,
              "phase": args.phase, "cases": {},
              "interpretation": "official reused development resets; not independent A1 replication"}
    reference_keys = None
    reference_first = None
    for case in CASES:
        directory = base / case
        assert directory.joinpath("exit-code.txt").read_text().strip() == "0", case
        for checksum_file in ("CONTRACT_SHA256SUMS.txt", "SHA256SUMS.txt"):
            subprocess.run(["sha256sum", "-c", checksum_file], cwd=directory,
                           check=True, stdout=subprocess.DEVNULL)
        command = shlex.split(directory.joinpath("command.txt").read_text())
        for name, value in (("--task_suite_name", "libero_spatial"),
                            ("--num_trials_per_task", str(per_task)),
                            ("--initial-state-offset", str(offset)),
                            ("--seed", "0"), ("--env-seed", "1"),
                            ("--seed-protocol", "paired")):
            assert flag(command, name) == value, (case, name)
        assert "--trace-actions" in command and "--trace-observations" in command
        assert ("--awq-recovery-lora-state" in command) == (case in ("C3", "LW"))
        if case in ("C0", "C3", "LW"):
            assert flag(command, "--weight-bits") == "2"
            assert flag(command, "--awq-candidate") == "w2-attention-primary-g64-stage-w4"
            assert flag(command, "--awq-w4-layers") == "8,9,10,11,12,13,14,15,20,21,22,23"
        else:
            assert flag(command, "--weight-bits") == "4"
            assert flag(command, "--awq-scope") == ("none" if case == "BF16" else "all")
        if case == "LW":
            assert Path(flag(command, "--awq-recovery-lora-state")).resolve() == train.joinpath("e2e_adapter_state.pt").resolve()

        log = next(directory.glob("EVAL-*.txt")).read_text()
        manifests = [json.loads(m.group(1)) for m in MANIFEST.finditer(log)]
        successes = [m.group(1) == "True" for m in SUCCESS.finditer(log)]
        assert len(manifests) == len(successes) == expected, (case, len(manifests), len(successes))
        keys = [(m["task_id"], m["init_state_index"], m["init_state_sha256"],
                 m["env_seed"], m["model_seed"], m["protocol"]) for m in manifests]
        assert {(m["task_id"], m["init_state_index"]) for m in manifests} == {
            (i, j) for i in range(10) for j in range(offset, offset + per_task)}
        assert len(set(keys)) == expected
        if reference_keys is None:
            reference_keys = keys
        else:
            assert keys == reference_keys, case

        events = lines(directory / "on-policy-events.jsonl")
        queries = lines(directory / "policy-queries.jsonl")
        event_queries = [row for row in events if row["record_type"] == "query"]
        assert sum(row["record_type"] == "episode_start" for row in events) == expected
        assert sum(row["record_type"] == "episode_end" for row in events) == expected
        assert len(event_queries) == len(queries) > 0
        assert all(row["chunk_execution_steps"] == 8 for row in event_queries)
        assert all(row["finite"] for row in queries)
        assert all(len(row["raw_policy_chunk"]) == 8 and all(len(a) == 7 and
                   all(math.isfinite(float(x)) for x in a) for a in row["raw_policy_chunk"])
                   for row in queries)
        first = {row["episode_serial"]: row["observation_sha256"] for row in event_queries
                 if row["query_in_episode"] == 0}
        assert set(first) == set(range(expected))
        if reference_first is None:
            reference_first = first
        else:
            assert first == reference_first, (case, "first policy observations differ")
        output["cases"][case] = {"successes": sum(successes), "success": successes,
                                 "query_count": len(queries),
                                 "command_sha256": digest(directory / "command.txt"),
                                 "log_sha256": digest(next(directory.glob("EVAL-*.txt"))),
                                 "events_sha256": digest(directory / "on-policy-events.jsonl"),
                                 "queries_sha256": digest(directory / "policy-queries.jsonl")}
    for case in ("C3", "LW"):
        base_success = output["cases"]["C0"]["success"]
        candidate_success = output["cases"][case]["success"]
        output["cases"][case]["rescue_vs_C0"] = sum(not a and b for a, b in zip(base_success, candidate_success))
        output["cases"][case]["break_vs_C0"] = sum(a and not b for a, b in zip(base_success, candidate_success))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"gate": output["gate"], "successes": {c: output["cases"][c]["successes"] for c in CASES},
                      "C3_rescue_break": [output["cases"]["C3"]["rescue_vs_C0"], output["cases"]["C3"]["break_vs_C0"]],
                      "LW_rescue_break": [output["cases"]["LW"]["rescue_vs_C0"], output["cases"]["LW"]["break_vs_C0"]]}))


if __name__ == "__main__":
    main()
