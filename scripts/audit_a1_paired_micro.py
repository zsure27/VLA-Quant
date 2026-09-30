"""Audit the fixed C0/C3 paired micro before extending the first shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def without_case_specific_flags(command: list[str]) -> list[str]:
    result = []
    index = 0
    while index < len(command):
        if command[index] in ("--local_log_dir", "--awq-recovery-lora-state"):
            index += 2
        else:
            result.append(command[index])
            index += 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.run_dir
    pairs = read_jsonl(base / "paired-episodes.jsonl")
    assert len(pairs) == 10
    assert {(row["task_id"], row["init_state_index"]) for row in pairs} == {(i, 5) for i in range(10)}
    assert all(row["C0"] in (True, False) and row["C3"] in (True, False) for row in pairs)
    assert all("env_seed" in row and "model_seed" in row and "init_state_sha256" in row for row in pairs)
    assert (base / "material_sha256_before.txt").read_bytes() == (base / "material_sha256_after.txt").read_bytes()
    summary = json.loads((base / "paired-summary.json").read_text(encoding="utf-8"))
    success = {case: sum(row[case] for row in pairs) for case in ("C0", "C3")}
    rescue = sum(not row["C0"] and row["C3"] for row in pairs)
    new_failure = sum(row["C0"] and not row["C3"] for row in pairs)
    assert summary["successes"] == success
    assert (summary["rescue"], summary["new_failure"]) == (rescue, new_failure)

    first_obs = {}
    first_action = {}
    command = {}
    query_counts = {}
    for case in ("C0", "C3"):
        directory = base / case
        assert (directory / "exit-code.txt").read_text().strip() == "0"
        assert json.loads((directory / "complete.json").read_text())["status"] == "COMPLETE"
        command[case] = shlex.split((directory / "command.txt").read_text())
        assert command[case][command[case].index("--env-seed") + 1] == "1"
        assert command[case][command[case].index("--initial-state-offset") + 1] == "5"
        assert command[case][command[case].index("--num_trials_per_task") + 1] == "1"
        assert ("--awq-recovery-lora-state" in command[case]) == (case == "C3")
        assert "--trace-actions" in command[case] and "--trace-observations" in command[case]
        events = read_jsonl(directory / "on-policy-events.jsonl")
        queries = read_jsonl(directory / "policy-queries.jsonl")
        event_queries = [row for row in events if row["record_type"] == "query"]
        assert len(event_queries) == len(queries)
        assert sum(row["record_type"] == "episode_start" for row in events) == 10
        assert sum(row["record_type"] == "episode_end" for row in events) == 10
        assert all(row["chunk_execution_steps"] == 8 for row in event_queries)
        assert all(row["finite"] for row in queries)
        first_obs[case] = {}
        first_action[case] = {}
        for event, query in zip(event_queries, queries):
            if event["query_in_episode"] == 0:
                serial = event["episode_serial"]
                first_obs[case][serial] = event["observation_sha256"]
                first_action[case][serial] = query["raw_policy_chunk"]
        assert set(first_obs[case]) == set(range(10))
        query_counts[case] = len(queries)

    assert without_case_specific_flags(command["C0"]) == without_case_specific_flags(command["C3"])
    equal_first_observations = sum(first_obs["C0"][i] == first_obs["C3"][i] for i in range(10))
    different_first_actions = sum(first_action["C0"][i] != first_action["C3"][i] for i in range(10))
    assert equal_first_observations == 10
    assert different_first_actions == 10
    output = {
        "plan_id": "a1-035-c0-c3-envseed1-micro-20260930",
        "gate": "PASS_POLICY_MICRO",
        "interpretation_limit": "ten historical-development official reset states under corrected env-seed protocol; no closed-loop benefit claim from this tiny slice",
        "pairs": 10,
        "successes": success,
        "rescue": rescue,
        "break": new_failure,
        "net_gain": success["C3"] - success["C0"],
        "first_observation_equal_pairs": equal_first_observations,
        "first_action_diff_pairs": different_first_actions,
        "policy_queries": query_counts,
        "all_queries_finite_and_eight_step_chunks": True,
        "material_sha256_before_after_identical": True,
        "cases_differ_only_by_adapter_and_output_path": True,
        "raw_paired_rows_sha256": sha(base / "paired-episodes.jsonl"),
        "raw_result_hashes": {case: {
            "on_policy_events": sha(base / case / "on-policy-events.jsonl"),
            "policy_queries": sha(base / case / "policy-queries.jsonl"),
        } for case in ("C0", "C3")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: output[key] for key in (
        "gate", "pairs", "successes", "rescue", "break",
        "first_observation_equal_pairs", "first_action_diff_pairs",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
