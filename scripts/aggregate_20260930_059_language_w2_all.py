"""Audit and aggregate 059 paired Spatial development resets 20–49."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
from pathlib import Path

CASES = ("C0", "C3", "LW", "BF16", "W4")
PHASES = (("first50trace", 20, 5), ("remaining250", 25, 25))
IDENTITY = ("task_id", "init_state_index", "init_state_sha256", "env_seed", "model_seed", "protocol")
MANIFEST = re.compile(r"EPISODE_MANIFEST (\{.*\})")
SUCCESS = re.compile(r"Success: (True|False)")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_case(directory: Path, count: int, start: int) -> tuple[list[tuple], list[bool], dict]:
    assert directory.joinpath("exit-code.txt").read_text().strip() == "0", directory
    for checksum_file in ("CONTRACT_SHA256SUMS.txt", "SHA256SUMS.txt"):
        subprocess.run(["sha256sum", "-c", checksum_file], cwd=directory,
                       check=True, stdout=subprocess.DEVNULL)
    logs = list(directory.glob("EVAL-*.txt"))
    assert len(logs) == 1, directory
    content = logs[0].read_text()
    manifests = [json.loads(m.group(1)) for m in MANIFEST.finditer(content)]
    successes = [m.group(1) == "True" for m in SUCCESS.finditer(content)]
    assert len(manifests) == len(successes) == count * 10, directory
    keys = [tuple(m[name] for name in IDENTITY) for m in manifests]
    assert len(set(keys)) == count * 10, directory
    assert {(m["task_id"], m["init_state_index"]) for m in manifests} == {
        (task, reset) for task in range(10) for reset in range(start, start + count)}
    assert all(m["protocol"] == "paired" for m in manifests)
    command = directory.joinpath("command.txt").read_text()
    for arg in ("--seed 0", "--env-seed 1", "--seed-protocol paired", "--trace-actions",
                f"--initial-state-offset {start}", f"--num_trials_per_task {count}"):
        assert arg in command, (directory, arg)
    assert len(directory.joinpath("policy-queries.jsonl").read_text().splitlines()) > 0
    return keys, successes, {"eval_log": sha(logs[0]), "command": sha(directory / "command.txt"),
                             "policy_queries": sha(directory / "policy-queries.jsonl")}


def mcnemar(rescue: int, new_failure: int) -> float:
    n = rescue + new_failure
    if not n:
        return 1.0
    low = min(rescue, new_failure)
    term = cumulative = 1
    for k in range(1, low + 1):
        term = term * (n - k + 1) // k
        cumulative += term
    return min(1.0, 2 * cumulative / 2**n)


def task_cluster_ci(rows: list[dict], candidate: str, baseline: str, draws: int = 20000) -> list[float]:
    tasks = [sum(int(row[candidate]) - int(row[baseline]) for row in rows
                 if row["task_id"] == task) for task in range(10)]
    rng = random.Random(20260930)
    values = [sum(rng.choice(tasks) for _ in range(10)) / 300 for _ in range(draws)]
    values.sort()
    return [values[int(0.025 * draws)], values[int(0.975 * draws)]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    provenance = {}
    for phase, start, count in PHASES:
        case_keys = {}
        case_success = {}
        provenance[phase] = {}
        for case in CASES:
            directory = args.root / f"eval-{phase}" / case
            case_keys[case], case_success[case], provenance[phase][case] = read_case(directory, count, start)
        assert all(case_keys[case] == case_keys["C0"] for case in CASES)
        for serial, identity in enumerate(case_keys["C0"]):
            row = dict(zip(IDENTITY, identity))
            row.update({case: case_success[case][serial] for case in CASES})
            rows.append(row)
    assert len(rows) == 300
    assert len({(r["task_id"], r["init_state_index"]) for r in rows}) == 300
    successes = {case: sum(int(r[case]) for r in rows) for case in CASES}
    by_task = {str(task): {case: sum(int(r[case]) for r in rows if r["task_id"] == task)
                           for case in CASES} for task in range(10)}
    comparisons = {}
    for candidate in ("C3", "LW"):
        rescue = sum(not r["C0"] and r[candidate] for r in rows)
        broken = sum(r["C0"] and not r[candidate] for r in rows)
        delta_by_task = {str(task): by_task[str(task)][candidate] - by_task[str(task)]["C0"]
                         for task in range(10)}
        comparisons[f"{candidate}_vs_C0"] = {
            "rescue": rescue, "break": broken, "net": rescue - broken,
            "exact_mcnemar_p": mcnemar(rescue, broken),
            "task_cluster_bootstrap_95ci_fraction": task_cluster_ci(rows, candidate, "C0"),
            "delta_by_task": delta_by_task,
            "tasks_improved": sum(x > 0 for x in delta_by_task.values()),
            "tasks_harmed": sum(x < 0 for x in delta_by_task.values())}
    output = {
        "classification": "reused official development resets20–49; not independent A1 or blind holdout",
        "episodes_per_case": 300, "complete_strict_pairing": True,
        "successes": successes, "by_task": by_task, "comparisons": comparisons,
        "same_condition_gap_to_BF16": successes["BF16"] - successes["LW"],
        "same_condition_gap_to_full_W4": successes["W4"] - successes["LW"],
        "provenance_sha256": provenance,
        "paired_rows_sha256": hashlib.sha256("\n".join(json.dumps(r, sort_keys=True) for r in rows).encode()).hexdigest(),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"successes": successes, "comparisons": comparisons,
                      "gaps": [output["same_condition_gap_to_BF16"],
                               output["same_condition_gap_to_full_W4"]]}))


if __name__ == "__main__":
    main()
