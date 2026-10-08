"""Register and audit the paired B2/B3 development-reset boundary slice."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(os.environ.get("QVLA_SOURCE_ROOT", str(Path(__file__).resolve().parent.parent))).resolve()
sys.path.insert(0, str(ROOT))
from scripts.audit_extended_peft_eval import audit_case
from qvla.paired_metrics import comparison

CASES = ("A3", "B1", "B2", "A4", "B3", "BF16", "A0")
OFFSET = 25
PHASES = (("micro", 1), ("first50", 5))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def replace_arg(argv: list[str], flag: str, value: str) -> None:
    index = argv.index(flag)
    argv[index + 1] = value


def build(args: argparse.Namespace) -> None:
    from scripts.vla_stage_supervisor import validate_plan

    if socket.gethostname() != args.expected_hostname:
        raise ValueError("Current host is not the registered 046 instance")
    if args.output.exists() or args.session.exists():
        raise ValueError("Immutable plan/session already exists")
    b2 = json.loads(args.b2_plan.read_text())
    b3 = json.loads(args.b3_plan.read_text())
    if b2["terminal_state"] != "AWAITING_GATE_REVIEW" or b3["terminal_state"] != "AWAITING_GATE_REVIEW":
        raise ValueError("Source plans are not gate-bounded")
    locks = [json.loads(path.read_text()) for path in (args.b2_lock, args.b3_lock)]
    files: dict[str, str] = {}
    for lock in locks:
        if lock["expected_hostname"] != args.expected_hostname:
            raise ValueError("Source lock host mismatch")
        for path, sha in lock["files"].items():
            if path in files and files[path] != sha:
                raise ValueError(f"Source locks disagree on {path}")
            files[path] = sha
    files[str(Path(__file__).resolve())] = digest(Path(__file__).resolve())
    args.output.mkdir(parents=True)
    args.session.mkdir(parents=True)
    lock_path = args.output / "boundary-lock.json"
    write_new(lock_path, {"version": "B2B3_DEV_BOUNDARY_V1", "expected_hostname": args.expected_hostname,
                          "files": files, "max_stage_seconds": 7200, "max_plan_seconds": 18000})
    stages = []
    for phase, count in PHASES:
        for case in CASES:
            original = next(stage for stage in (b2 if case in ("A3", "B1", "B2") else b3)["stages"]
                            if stage["id"] == f"first50-{case}")
            stage = copy.deepcopy(original)
            stage["id"] = f"{phase}-{case}"
            command = stage["command"]
            replace_arg(command, "--lock", str(lock_path))
            replace_arg(command, "--output", str(args.session / ("eval-" + phase) / case))
            replace_arg(command, "--initial-state-offset", str(OFFSET))
            replace_arg(command, "--num_trials_per_task", str(count))
            replace_arg(command, "--local_log_dir", str(args.session / ("eval-" + phase) / case))
            stages.append(stage)
        template = copy.deepcopy(stages[-1])
        template["id"] = "audit-" + phase
        template["env"]["QVLA_SOURCE_ROOT"] = str(ROOT)
        template["command"] = [args.python, str(ROOT / "scripts/run_locked_peft_stage.py"),
                               "--lock", str(lock_path), "--output", str(args.session / "stage-records" / template["id"]),
                               "--", args.python, str(Path(__file__).resolve()), "audit", "--session", str(args.session),
                               "--phase", phase, "--output", str(args.session / f"{phase}-protocol-gate.json")]
        stages.append(template)
    plan = {"schema_version": "1.0", "plan_id": args.plan_id,
            "control_root": str(args.output / "control"), "terminal_state": "AWAITING_GATE_REVIEW",
            "stages": stages, "question": "Development reset25–29 boundary: B2 vs B1 and B3 vs A4",
            "classification": "reused official development resets; not independent replication",
            "primary_variables": ["visual rank8 LoRA conditional on frozen B1", "language rank8 LoRA conditional on A4"],
            "next_expansion": "LOCKED until first50 protocol audit and new preregistration"}
    validate_plan(plan)
    write_new(args.output / "boundary-plan.json", plan)
    write_new(args.output / "preregistration.json", {"plan_id": args.plan_id,
        "source_plan_sha256": {"B2": digest(args.b2_plan), "B3": digest(args.b3_plan)},
        "source_lock_sha256": {"B2": digest(args.b2_lock), "B3": digest(args.b3_lock)},
        "audit_script_sha256": digest(Path(__file__).resolve()),
        "train_resets": [0, 1, 2, 3], "eval_resets": list(range(25, 30)),
        "cases": CASES, "phases": [{"name": p, "offset": OFFSET, "per_task": n} for p, n in PHASES],
        "micro_overlaps_first50": True, "first50_per_case": 50,
        "continuation_gate": "protocol validity only; success direction never gates continuation"})
    print(json.dumps({"gate": "PLAN_REGISTERED_NOT_STARTED", "plan": str(args.output / "boundary-plan.json")}))


def audit(args: argparse.Namespace) -> None:
    count = dict(PHASES)[args.phase]
    results = {case: audit_case(args.session / ("eval-" + args.phase) / case, case, OFFSET, count)
               for case in CASES}
    reference = results["A4"]
    if any(value["keys"] != reference["keys"] or value["first"] != reference["first"] for value in results.values()):
        raise ValueError("Paired keys or first policy-visible observations differ")
    rows = [{"task_id": key[0], "init_state_index": key[1],
             **{case: results[case]["success"][i] for case in CASES}}
            for i, key in enumerate(reference["keys"])]
    output = {"gate": "PASS_PROTOCOL_BOUNDARY_" + args.phase.upper(),
              "classification": "reused official development resets, not independent replication",
              "offset": OFFSET, "per_task": count, "paired_episodes": len(rows), "paired_rows": rows,
              "cases": {case: {"successes": sum(results[case]["success"]), "episodes": len(rows),
                               "query_count": results[case]["query_count"], "sha256": results[case]["sha256"]}
                        for case in CASES},
              "B2_vs_B1": comparison(rows, "B2", "B1"),
              "B3_vs_A4": comparison(rows, "B3", "A4"),
              "B3_vs_B1": comparison(rows, "B3", "B1"),
              "B2_vs_B3": comparison(rows, "B2", "B3"),
              "next_expansion": "LOCKED; protocol-only audit and new preregistration required"}
    write_new(args.output, output)
    print(json.dumps({"gate": output["gate"], "B2_vs_B1": output["B2_vs_B1"],
                      "B3_vs_A4": output["B3_vs_A4"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    for name in ("b2-plan", "b3-plan", "b2-lock", "b3-lock", "output", "session"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--expected-hostname", required=True)
    p.add_argument("--plan-id", required=True)
    p.add_argument("--python", required=True)
    p = sub.add_parser("audit")
    p.add_argument("--session", required=True, type=Path)
    p.add_argument("--phase", required=True, choices=[p for p, _ in PHASES])
    p.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    (build if args.command == "build" else audit)(args)


if __name__ == "__main__":
    main()
