"""Strict paired protocol/initial-factor role audit for 091 M0 micro or first50."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qvla.b3_init_only_091_v1 import MODELS, VERSION, sha, validate_provenance


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_first_query_pair(cases):
    """Numerical role checks separate from rollout labels; useful for CPU tests."""
    import numpy as np
    expected_parameters = {"ptq": 0, "svd-init-only": 19988480}
    for variant, case in cases.items():
        candidate, gate, query = case["candidate"], case["actual_gate"], case["first_query"]
        if (candidate.get("version") != VERSION or candidate.get("candidate") != variant
                or candidate.get("model_id") != MODELS[variant]
                or candidate.get("base_model_id") != "A4"
                or candidate.get("optimization_steps") != 0
                or candidate.get("training_executed") is not False
                or candidate.get("parameters") != expected_parameters[variant]
                or candidate.get("rank") != (8 if variant == "svd-init-only" else 0)
                or candidate.get("initial_state_used") is not (variant == "svd-init-only")):
            raise ValueError("Candidate is not the registered 0-step PTQ/S0 role")
        if (gate.get("gate") != "PASS_B3_INIT_ONLY_ACTUAL_MODEL"
                or gate.get("version") != VERSION or gate.get("candidate") != variant
                or gate.get("optimization_steps") != 0
                or gate.get("reload_output_equal") is not True
                or gate.get("all_parameters_frozen") is not True
                or gate.get("single_proprio_normalization_equal") is not True
                or gate.get("state_space") != "raw_proprio_before_get_action"):
            raise ValueError("Actual-query frozen/proprio/reload gate incomplete")
        if (query.get("state_space") != "raw_proprio_before_get_action"
                or not np.array_equal(np.asarray(gate["raw_state"]), np.asarray(query["state"]))):
            raise ValueError("First-query gate does not bind the actual raw proprio")
        reference, executed = np.asarray(gate["reference_action"]), np.asarray(query["raw_policy_chunk"])
        zero = np.asarray(gate["zero_branch_action"])
        if (reference.shape != (8, 7) or zero.shape != (8, 7)
                or not np.isfinite(reference).all() or not np.isfinite(zero).all()
                or not np.array_equal(reference, executed)):
            raise ValueError("Actual first action differs from admitted/reloaded candidate output")
    ptq, s0 = cases["ptq"], cases["svd-init-only"]
    for field in ("checkpoint_identity", "profiles", "parent_artifact_sha256", "initial_state_sha256", "sources"):
        if ptq["candidate"].get(field) != s0["candidate"].get(field):
            raise ValueError(f"Uncontrolled PTQ/S0 provenance difference: {field}")
    if (ptq["actual_gate"]["frozen_sha256"] != s0["actual_gate"]["frozen_sha256"]
            or not np.array_equal(np.asarray(s0["actual_gate"]["zero_branch_action"]),
                                  np.asarray(ptq["first_query"]["raw_policy_chunk"]))
            or not np.array_equal(np.asarray(ptq["actual_gate"]["zero_branch_action"]),
                                  np.asarray(ptq["first_query"]["raw_policy_chunk"]))):
        raise ValueError("S0 zero-branch action is not exactly canonical PTQ on the paired first observation")


def audit(root, offset, count):
    from scripts.audit_extended_peft_eval import audit_case, read_lines
    from qvla.reproducibility import episode_seeds
    from qvla.paired_metrics import comparison
    from qvla.run_eval_b3_init_only_091_v1 import parse_args
    if offset != 5 or count not in (1, 5):
        raise ValueError("M0 admission is only reset5 micro or reset5-9 first50")
    records, cases, costs = {}, {}, {}
    current_sources = {name: sha(ROOT / name) for name in
        ("qvla/run_eval_b3_init_only_091_v1.py", "qvla/b3_init_only_091_v1.py")}
    for variant, model_id in MODELS.items():
        directory = root / variant
        invocation = read_json(directory / "invocation.json")
        command = invocation["command"]
        expected_entry = "/qvla/run_eval_b3_init_only_091_v1.py"
        entries = [i for i, item in enumerate(command) if str(item).replace("\\", "/").endswith(expected_entry)]
        if len(entries) != 1:
            raise ValueError("Actual command did not use the new independent S0 evaluator")
        args = parse_args(command[entries[0] + 1:])
        if args.candidate != variant or args.local_log_dir.resolve() != directory.resolve():
            raise ValueError("Actual candidate/output differs from phase registration")
        parent = validate_provenance(args)
        records[variant] = audit_case(directory, model_id, offset, count)
        log = next(directory.glob("EVAL-*.txt")).read_text(encoding="utf-8")
        console = (directory / "console.log").read_text(encoding="utf-8", errors="replace")
        if re.search(r"Episode error:|Traceback \(most recent call last\)", log + console):
            raise ValueError("Completed case contains a program exception")
        for key in records[variant]["keys"]:
            model_seed, env_seed = episode_seeds(0, 1, "libero_spatial", key[0], key[1])
            if key[3:] != (env_seed, model_seed, "paired"):
                raise ValueError("Actual derived episode seeds differ from the pinned paired protocol")
        first_query = read_lines(directory / "policy-queries.jsonl")[0]
        candidate = read_json(directory / "candidate-init-only.json")
        gate = read_json(directory / "actual-model-init-only-gate.json")
        if (candidate.get("sources") != current_sources
                or candidate.get("parent_artifact_sha256") != args.initial_artifact_sha256
                or candidate.get("initial_state_sha256") != sha(args.initial_state)
                or candidate.get("checkpoint_identity") != parent["checkpoint_identity"]
                or gate.get("frozen_sha256") != parent["frozen_before_sha256"]):
            raise ValueError("Candidate/gate does not bind the actual admitted source/parent/factor/base")
        cases[variant] = {"candidate": candidate, "actual_gate": gate,
                          "first_query": first_query, "invocation": invocation}
        started = invocation.get("started_unix")
        finished = (directory / "exit-code.txt").stat().st_mtime
        wall = finished - started if isinstance(started, (int, float)) else None
        if wall is None or wall < 0:
            raise ValueError("Missing or invalid wrapper timing evidence")
        costs[variant] = {"policy_queries": records[variant]["query_count"],
            "wrapper_wall_seconds": wall,
            "wall_scope": "invocation.started_unix to exit-code mtime; includes model/checksum/first-query gates/trace/video/IO",
            "optimization_steps": 0, "new_teacher_labels": 0,
            "adapter_parameters": candidate["parameters"],
            "adapter_tensor_bytes": 39976960 if variant == "svd-init-only" else 0,
            "adapter_serialized_bytes": args.initial_state.stat().st_size if variant == "svd-init-only" else 0,
            "initialization_cost": "S0 reuses prior calibration/SVD factors; historical construction cost is additional, not zero",
            "interpretation": "query/wall differences can reflect episode length; no kernel/packed-low-bit speed claim"}
    ptq, s0 = records["ptq"], records["svd-init-only"]
    if ptq["keys"] != s0["keys"] or ptq["first"] != s0["first"]:
        raise ValueError("PTQ/S0 task/reset/seed/initial observation pairing differs")
    for field in ("hostname", "lock_sha256"):
        if not cases["ptq"]["invocation"].get(field) or cases["ptq"]["invocation"][field] != cases["svd-init-only"]["invocation"].get(field):
            raise ValueError("PTQ/S0 run is not from the same admitted host/runtime lock")
    validate_first_query_pair(cases)
    rows = [{"task_id": key[0], "init_state_index": key[1],
             "ptq": ptq["success"][index], "svd-init-only": s0["success"][index]}
            for index, key in enumerate(ptq["keys"])]
    failures = {variant: [{"task_id": row["task_id"], "init_state_index": row["init_state_index"]}
                         for row in rows if not row[variant]] for variant in MODELS}
    return {"gate": "PASS_B3_INIT_ONLY_PAIRED_PROTOCOL", "version": VERSION,
        "phase": "micro" if count == 1 else "first50", "paired_episodes": len(rows),
        "offset": offset, "count_per_task": count,
        "classification": "same-task development; micro overlaps formal and must not be added",
        "paired_rows": rows, "comparison_S0_minus_PTQ": comparison(rows, "svd-init-only", "ptq"),
        "successes": {variant: sum(row[variant] for row in rows) for variant in MODELS},
        "rescue_pairs": [row for row in rows if row["svd-init-only"] and not row["ptq"]],
        "break_pairs": [row for row in rows if row["ptq"] and not row["svd-init-only"]],
        "failures": failures, "costs": costs,
        "zero_branch_vs_PTQ_first_action_equal": True,
        "provenance": {variant: {"audit_sha256": records[variant]["sha256"],
            "candidate_sha256": sha(root / variant / "candidate-init-only.json"),
            "actual_model_gate_sha256": sha(root / variant / "actual-model-init-only-gate.json"),
            "parent_artifact_sha256": cases[variant]["candidate"]["parent_artifact_sha256"]}
            for variant in MODELS},
        "next_gate": "Protocol/finite/exception/identity admission only; no success-rate-dependent continuation",
        "limits": "Old Z/S 450 remains frozen historical evidence; this gate has no new training or independent-condition claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--count", type=int, choices=(1, 5), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Gate already exists; do not overwrite")
    result = audit(args.root, args.offset, args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in
        ("gate", "phase", "paired_episodes", "successes", "comparison_S0_minus_PTQ")}))


if __name__ == "__main__":
    main()
