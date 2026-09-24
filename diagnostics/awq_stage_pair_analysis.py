"""Audit paired AWQ stage-pruning batches from frozen closed-loop outputs."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_manifests(text: str) -> list[dict]:
    lines = text.splitlines()
    manifests = []
    for index, line in enumerate(lines):
        if "EPISODE_MANIFEST" not in line:
            continue
        pieces = []
        started = False
        for candidate in lines[index + 1 :]:
            value = candidate.strip()
            if not started:
                brace = value.find("{")
                if brace < 0:
                    continue
                value = value[brace:]
                started = True
            pieces.append(value)
            if "}" in value:
                break
        if not pieces or "}" not in pieces[-1]:
            raise ValueError("incomplete EPISODE_MANIFEST")
        manifests.append(json.loads(re.sub(r"\s+", "", "".join(pieces))))
    return manifests


def analyze_result(directory: Path) -> dict:
    complete = json.loads((directory / "complete.json").read_text())
    if complete.get("status") != "EXECUTION_COMPLETE":
        raise ValueError(f"incomplete result: {directory}")
    exits = [path.read_text().strip() for path in directory.rglob("exit-code.txt")]
    if not exits or any(code != "0" for code in exits):
        raise ValueError(f"nonzero or missing exit code: {directory}")
    consoles = list(directory.glob("*/console.log"))
    traces = list(directory.glob("*/policy-queries.jsonl"))
    if len(consoles) != 1 or len(traces) != 1:
        raise ValueError(f"expected one console and trace: {directory}")
    text = consoles[0].read_text(errors="replace")
    episodes = int(re.findall(r"Total episodes:\s*(\d+)", text)[-1])
    successes = int(re.findall(r"Total successes:\s*(\d+)", text)[-1])
    outcomes = [value == "True" for value in re.findall(r">> Success:\s+(True|False)", text)]
    manifests = parse_manifests(text)
    if len(outcomes) != episodes or len(manifests) != episodes:
        raise ValueError(f"episode evidence count mismatch: {directory}")
    queries = 0
    for line in traces[0].open():
        record = json.loads(line)
        chunk, state = record["raw_policy_chunk"], record["state"]
        if record.get("finite") is not True or len(chunk) != 8 or any(len(row) != 7 for row in chunk):
            raise ValueError(f"invalid action trace shape: {traces[0]}")
        if len(state) != 8 or record.get("state_space") != "raw_proprio_before_get_action":
            raise ValueError(f"invalid proprio trace: {traces[0]}")
        if not all(math.isfinite(float(value)) for row in chunk for value in row + state):
            raise ValueError(f"non-finite trace: {traces[0]}")
        queries += 1
    if queries == 0:
        raise ValueError(f"empty trace: {traces[0]}")
    return {
        "directory": str(directory),
        "episodes": episodes,
        "successes": successes,
        "outcomes": outcomes,
        "manifests": manifests,
        "policy_queries": queries,
        "trace_sha256": sha256(traces[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    batches = []
    for batch in args.batch:
        runs = [Path(line) for line in (batch / "RUNS.txt").read_text().splitlines() if line]
        if len(runs) != 2:
            raise ValueError(f"expected two paired runs: {batch}")
        reference, pruned = map(analyze_result, runs)
        if reference["manifests"] != pruned["manifests"]:
            raise ValueError(f"paired manifests differ: {batch}")
        batches.append({
            "batch": str(batch),
            "episodes_per_run": reference["episodes"],
            "reference_successes": reference["successes"],
            "pruned_successes": pruned["successes"],
            "paired_both_success": sum(a and b for a, b in zip(reference["outcomes"], pruned["outcomes"])),
            "paired_reference_only": sum(a and not b for a, b in zip(reference["outcomes"], pruned["outcomes"])),
            "paired_pruned_only": sum(not a and b for a, b in zip(reference["outcomes"], pruned["outcomes"])),
            "paired_both_fail": sum(not a and not b for a, b in zip(reference["outcomes"], pruned["outcomes"])),
            "reference_policy_queries": reference["policy_queries"],
            "pruned_policy_queries": pruned["policy_queries"],
            "reference_trace_sha256": reference["trace_sha256"],
            "pruned_trace_sha256": pruned["trace_sha256"],
            "manifest_sha256": hashlib.sha256(json.dumps(reference["manifests"], sort_keys=True).encode()).hexdigest(),
            "result_directories": [reference["directory"], pruned["directory"]],
        })
    result = {
        "status": "AUDIT_COMPLETE",
        "batches": batches,
        "total_episodes_per_candidate": sum(item["episodes_per_run"] for item in batches),
        "reference_successes": sum(item["reference_successes"] for item in batches),
        "pruned_successes": sum(item["pruned_successes"] for item in batches),
        "all_manifests_paired": True,
        "all_traces_finite_8x7_with_proprio": True,
    }
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
