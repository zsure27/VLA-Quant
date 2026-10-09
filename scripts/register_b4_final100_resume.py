"""Register a one-time B4 final100 recovery without repeating completed B3."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import socket

from qvla.b4_joint_peft import SOURCES
from qvla.extended_peft import sha
from scripts.vla_stage_supervisor import paths, validate_plan


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-plan", required=True, type=Path)
    p.add_argument("--new-checkout", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--expected-hostname", required=True)
    a = p.parse_args()
    if socket.gethostname() != a.expected_hostname or a.output.exists():
        raise ValueError("Wrong instance or recovery plan already exists")
    old = json.loads(a.old_plan.read_text())
    _, status_path, _, _ = paths(a.old_plan, old)
    status = json.loads(status_path.read_text())
    if (status["state"] != "FAILED" or status["current_stage"] != "final100-B4" or
            "final100-B3" not in status["completed_stage_ids"]):
        raise ValueError("Only the diagnosed B4-only failure can be recovered")
    session = Path(old["stages"][0]["command"][old["stages"][0]["command"].index("--root") + 1])
    if not (session / "eval-final100/B3/exit-code.txt").read_text().strip() == "0":
        raise ValueError("Completed B3 final100 must remain intact")
    failed = session / "eval-final100/B4"
    expected = {"console.log", "exit-code.txt", "invocation.json"}
    if {f.name for f in failed.iterdir()} != expected or (failed / "exit-code.txt").read_text().strip() == "0":
        raise ValueError("Failed B4 stub has unexpected data; refuse relocation")
    failed_copy = session / "failed-attempts/final100-B4-v4"
    if failed_copy.exists():
        raise ValueError("Failed-attempt archive already exists")
    old_checkout = Path(old["stages"][0]["cwd"])
    new_checkout = a.new_checkout.resolve()
    for relative in SOURCES:
        if sha(old_checkout / relative) != sha(new_checkout / relative):
            raise ValueError(f"Trained source changed: {relative}")
    wrapper = new_checkout / "scripts/eval_b4_final100_compat.py"
    registrar = new_checkout / "scripts/register_b4_final100_resume.py"
    if not wrapper.is_file() or not registrar.is_file():
        raise ValueError("Missing versioned recovery code")
    old_lock = Path(old["stages"][0]["command"][old["stages"][0]["command"].index("--lock") + 1])
    lock = json.loads(old_lock.read_text())
    for relative in SOURCES:
        lock["files"][str(new_checkout / relative)] = sha(new_checkout / relative)
    for path in (wrapper, registrar):
        lock["files"][str(path)] = sha(path)
    new_lock = a.output / "B4-runtime-lock-resume.json"
    plan = deepcopy(old)
    plan["plan_id"] = old["plan_id"] + "-resume-v1"
    plan["control_root"] = str(a.output / "control")
    plan["question"] = "Complete only the preregistered B4 reset40-49 after evaluator admission bug"
    chosen = {stage["id"]: deepcopy(stage) for stage in old["stages"]}
    plan["stages"] = [chosen[key] for key in ("verify-last100", "final100-B4", "audit-final100", "audit-combined")]
    for stage in plan["stages"]:
        original_id = stage["id"]
        if original_id == "final100-B4":
            stage["id"] = "final100-B4-resume"
        stage["cwd"] = str(new_checkout)
        stage["env"]["PYTHONPATH"] = stage["env"]["PYTHONPATH"].replace(str(old_checkout), str(new_checkout))
        command = [value.replace(str(old_checkout), str(new_checkout)) for value in stage["command"]]
        command[command.index("--lock") + 1] = str(new_lock)
        if original_id == "final100-B4":
            command = [value.replace("qvla/run_eval_b4_joint.py", "scripts/eval_b4_final100_compat.py") for value in command]
        else:
            command[command.index("--output") + 1] = str(session / "stage-records/resume-v1" / original_id)
        stage["command"] = command
    validate_plan(plan)
    a.output.mkdir(parents=True)
    new_lock.write_text(json.dumps(lock, indent=2) + "\n")
    (a.output / "B4-final100-resume-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    failed_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(failed), str(failed_copy))
    (failed_copy / "RELOCATION.json").write_text(json.dumps({
        "reason": "B4 reset40-49 admission rejected before model/evaluation startup",
        "original": str(failed), "files_sha256": {name: sha(failed_copy / name) for name in sorted(expected)},
        "original_plan_sha256": status["plan_sha256"]}, indent=2) + "\n")
    print(json.dumps({"gate": "PASS_B4_FINAL100_RECOVERY_REGISTERED", "B3_repeated": False,
                      "failed_attempt_preserved": str(failed_copy), "plan": str(a.output / "B4-final100-resume-plan.json")}))


if __name__ == "__main__":
    main()
