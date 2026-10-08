"""Protocol-only micro/first50 gate; report paired metrics without selection by success."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import re
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qvla.extended_peft import sha
from qvla.on_policy_capture import observation_hash
from qvla.paired_metrics import comparison


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def normalized_trace_state(raw, stats):
    """Mirror the frozen OFT BOUNDS_Q99 proprio transform, then NPZ float32 storage."""
    value = np.asarray(raw)
    q01, q99 = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    mask = np.asarray(stats.get("mask", np.ones_like(q01, dtype=bool)), dtype=bool)
    if value.shape != (8,) or q01.shape != (8,) or q99.shape != (8,) or mask.shape != (8,) or not np.isfinite(value).all():
        raise ValueError("Malformed raw trace or frozen proprio statistics")
    return np.clip(np.where(mask, 2 * (value - q01) / (q99 - q01 + 1e-8) - 1, value), -1, 1).astype(np.float32)


def audit_case(directory, model_id, offset, count):
    if (directory / "exit-code.txt").read_text().strip() != "0": raise ValueError("Evaluation failed")
    command = json.loads((directory / "invocation.json").read_text())["command"]
    def flag(key): return command[command.index(key) + 1]
    for key, value in (("--model-id", model_id), ("--initial-state-offset", str(offset)),
                       ("--num_trials_per_task", str(count)), ("--seed", "0"), ("--env-seed", "1"),
                       ("--seed-protocol", "paired"), ("--task_suite_name", "libero_spatial")):
        if flag(key) != value: raise ValueError(f"Actual command differs: {model_id} {key}")
    if "--trace-observations" not in command or "--trace-actions" not in command:
        raise ValueError("Both traces must really be enabled")
    statistics = Path(flag("--pretrained_checkpoint")) / "dataset_statistics.json"
    norm = json.loads(statistics.read_text(encoding="utf-8"))["libero_spatial_no_noops"]["proprio"]
    logs = list(directory.glob("EVAL-*.txt"))
    if len(logs) != 1: raise ValueError("Expected exactly one evaluation log")
    log = logs[0].read_text()
    manifests = [json.loads(m.group(1)) for m in re.finditer(r"EPISODE_MANIFEST (\{.*\})", log)]
    successes = [m.group(1) == "True" for m in re.finditer(r"Success: (True|False)", log)]
    expected = {(t, r) for t in range(10) for r in range(offset, offset + count)}
    keys = [(m["task_id"], m["init_state_index"], m["init_state_sha256"], m["env_seed"], m["model_seed"], m["protocol"]) for m in manifests]
    if len(keys) != len(expected) or len(successes) != len(expected) or len(set(keys)) != len(expected) or {(k[0], k[1]) for k in keys} != expected:
        raise ValueError("Completed episode count or exact task/reset coverage mismatch")
    events = read_lines(directory / "on-policy-events.jsonl")
    queries = read_lines(directory / "policy-queries.jsonl")
    eq = [r for r in events if r["record_type"] == "query"]
    starts = [r for r in events if r["record_type"] == "episode_start"]
    ends = [r for r in events if r["record_type"] == "episode_end"]
    if ({r["episode_serial"] for r in starts} != set(range(len(keys))) or len(starts) != len(keys)
            or len(ends) != len(keys) or [r["episode_serial"] for r in ends] != list(range(len(keys)))
            or [r["success"] for r in ends] != successes or any(r["aborted"] for r in ends)):
        raise ValueError("Episode trace incomplete/aborted or success mapping differs")
    if len(eq) != len(queries) or not eq: raise ValueError("Action/observation query traces mismatch")
    first, next_query = {}, {i: 0 for i in range(len(keys))}
    for row, query in zip(eq, queries):
        serial = row["episode_serial"]
        if serial not in next_query or row["query_in_episode"] != next_query[serial] or row["chunk_execution_steps"] != 8:
            raise ValueError("Non-contiguous query or wrong chunk execution semantics")
        if row["executed_step_start"] != row["query_in_episode"] * 8:
            raise ValueError("Re-inference must follow eight executed steps")
        next_query[serial] += 1
        action = np.asarray(query["raw_policy_chunk"])
        if query["finite"] is not True or action.shape != (8, 7) or not np.isfinite(action).all():
            raise ValueError("Nonfinite/non-7D/non-eight-step action")
        source = directory / row["file"]
        if not source.resolve().is_relative_to(directory.resolve()) or sha(source) != row["file_sha256"]:
            raise ValueError("Observation file missing or SHA mismatch")
        with np.load(source, allow_pickle=False) as sample:
            fingerprint = observation_hash(sample["image"], sample["wrist_image"], sample["state"], str(sample["instruction"].item()))
            if not np.array_equal(sample["student_action"], action.astype(np.float32)):
                raise ValueError("Saved observation action differs from execution trace")
            if (row["task"] != query["task"] or row["task"] != str(sample["instruction"].item())
                    or query.get("state_space") != "raw_proprio_before_get_action"
                    or not np.array_equal(sample["state"], normalized_trace_state(query["state"], norm))):
                raise ValueError("Trace task/proprio differs from saved policy observation")
        if fingerprint != row["observation_sha256"]: raise ValueError("Observation fingerprint mismatch")
        if row["query_in_episode"] == 0: first[serial] = fingerprint
    if set(first) != set(range(len(keys))) or any(r["queries"] != next_query[r["episode_serial"]] for r in ends):
        raise ValueError("Missing first observation or query count mismatch")
    return {"keys": keys, "success": successes, "first": first, "query_count": len(eq),
            "sha256": {p.name: sha(p) for p in (logs[0], directory / "invocation.json",
                        directory / "policy-queries.jsonl", directory / "on-policy-events.jsonl", statistics)}}


def audit(root, model_id, phase):
    controls = ("B1", "B2", "A3", "BF16", "A0") if model_id == "B2" else ("A4", "B3", "B1", "BF16", "A0")
    count = 1 if phase == "micro" else 5
    results = {c: audit_case(root / f"eval-{phase}" / c, c, 20, count) for c in controls}
    reference = results[controls[0]]
    if any(r["keys"] != reference["keys"] or r["first"] != reference["first"] for r in results.values()):
        raise ValueError("Policies differ in pairing keys or first policy-visible observations")
    rows = [{"task_id": key[0], "init_state_index": key[1], **{c: results[c]["success"][i] for c in controls}}
            for i, key in enumerate(reference["keys"])]
    output = {"gate": f"PASS_PROTOCOL_{phase.upper()}", "classification": "reused development resets, not independent replication",
        "model_id": model_id, "phase": phase, "paired_episodes": len(rows), "paired_rows": rows,
        "cases": {c: {"successes": sum(results[c]["success"]), "episodes": len(rows),
                      "query_count": results[c]["query_count"], "sha256": results[c]["sha256"]} for c in controls},
        "primary": comparison(rows, model_id, controls[0]),
        "references": {c: comparison(rows, model_id, c) for c in controls if c != model_id},
        "first50_expansion": "protocol-only; no success-rate gate",
        "remaining_long_expansion": "LOCKED; independent condition or new development preregistration required"}
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--model-id", required=True, choices=("B2", "B3"))
    p.add_argument("--phase", required=True, choices=("micro", "first50"))
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    result = audit(a.root, a.model_id, a.phase)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists(): raise ValueError("Gate exists; do not overwrite")
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"gate": result["gate"], "primary": result["primary"]}))
