"""Register immutable B3/B4 300-episode plans after CPU data and closure gates; never launch GPU."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.prepare_b4_pair_v2_plans import build_plans as build_v2
from scripts.audit_b4_data_scope import audit_materials, sha
from scripts.train_b4_joint_peft import materials_contract


def _replace_audit(stage: dict) -> dict:
    stage = deepcopy(stage)
    stage["command"] = [arg.replace("audit_b4_pair_v2.py", "audit_b4_pair_v3.py") for arg in stage["command"]]
    return stage


def _phase_stage(stage: dict, old: str, new: str) -> dict:
    stage = _replace_audit(stage)
    stage["id"] = stage["id"].replace(old, new)
    stage["command"] = [arg.replace(old, new) if isinstance(arg, str) else arg
                        for arg in stage["command"]]
    command = stage["command"]
    for flag in ("--phase", "--require-prior"):
        if flag in command and command[command.index(flag) + 1] == old:
            command[command.index(flag) + 1] = new
    if "--initial-state-offset" in command:
        command[command.index("--initial-state-offset") + 1] = "40"
    return stage


def build_plans(materials: dict, session: Path, output: Path, python: str, hostname: str) -> dict:
    plans = build_v2(materials, session, output, python, hostname)
    for phase, plan in plans.items():
        plan["plan_id"] = plan["plan_id"].replace("pair-v2", "pair-v3")
        plan["question"] = "Fresh B4-B3 paired 300 development episodes on reset20-49"
        plan["stages"] = [_replace_audit(stage) for stage in plan["stages"]]
        plan["next_plan"] = "Protocol audit then next immutable phase; final100 ends this run"
    last = plans["last100"]
    combined_stage = next(stage for stage in last["stages"] if stage["id"] == "audit-combined")
    last["stages"] = [stage for stage in last["stages"] if stage["id"] != "audit-combined"]
    previous = next(stage for stage in last["stages"] if stage["id"] == "verify-next50")
    evals = [stage for stage in last["stages"] if stage["id"] in ("last100-B3", "last100-B4")]
    audit = next(stage for stage in last["stages"] if stage["id"] == "audit-last100")
    final = deepcopy(last)
    final["plan_id"] = final["plan_id"].replace("last100", "final100")
    final["stages"] = [
        _phase_stage(previous, "next50", "last100"),
        *(_phase_stage(stage, "last100", "final100") for stage in evals),
        _phase_stage(audit, "last100", "final100"),
        _replace_audit(combined_stage),
    ]
    final["next_plan"] = "Analyze paired 300; archive, sync GitHub and close instance"
    plans["final100"] = final
    return plans


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("materials", "session", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--expected-hostname", required=True)
    a = p.parse_args()
    if socket.gethostname() != a.expected_hostname:
        raise ValueError("Wrong current instance")
    if a.session.exists() or a.output.exists():
        raise ValueError("Recover immutable existing plan; do not overwrite")
    materials = json.loads(a.materials.read_text(encoding="utf-8"))
    materials_contract(materials, "B4")
    receipt = audit_materials(materials)
    if receipt.get("gate") != "PASS_DATA_SCOPE":
        raise ValueError("CPU data content audit did not pass")
    for name in ("backup_active_diagnostics.py", "vla_shutdown_remote.py"):
        if sha(Path(materials["closure_tools"]) / name) != sha(ROOT / "scripts" / name):
            raise ValueError("Sync closure tools first")
    for key in ("oft_root", "libero_root", "official_root"):
        if not Path(materials[key]).is_dir():
            raise ValueError("Missing runtime")
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                         text=True, capture_output=True, check=False)
    if gpu.returncode == 0 and gpu.stdout.strip():
        raise ValueError("GPU busy; recover current plan")
    if gpu.returncode != 0:
        message = (gpu.stdout + gpu.stderr).lower()
        if "no devices were found" not in message and "no devices found" not in message:
            raise ValueError("Cannot distinguish GPU absence from an unexpected GPU error")
        listing = subprocess.run(["nvidia-smi", "-L"], text=True, capture_output=True, check=False)
        if listing.returncode != 0 or listing.stdout.strip().lower() != "no devices found.":
            raise ValueError("GPU absence check is inconsistent; do not register")
    if shutil.disk_usage(a.session.parent).free < 12 * 1024**3:
        raise ValueError("Require 12 GiB free with archive reserve")
    a.session.mkdir(parents=True)
    a.output.mkdir(parents=True)
    receipt_path = a.session / "data-scope-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plans = build_plans(materials, a.session, a.output, a.python, a.expected_hostname)
    lock_path = a.output / "B4-runtime-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for path in (receipt_path, ROOT / "scripts/prepare_b4_pair_v3_plans.py",
                 ROOT / "scripts/audit_b4_pair_v3.py", ROOT / "scripts/prepare_b4_pair_v2_plans.py",
                 ROOT / "scripts/audit_b4_pair_v2.py", ROOT / "scripts/audit_b4_data_scope.py",
                 ROOT / "configs/experiments/b4_eval_pair_v3_20261009.json"):
        lock["files"][str(path)] = sha(path)
    lock["files"].update(receipt["source_sha256"])
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    from scripts.vla_stage_supervisor import validate_plan
    for phase, plan in plans.items():
        validate_plan(plan)
        (a.output / f"B4-{phase}-plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gate": "PASS_B4_PAIR_V3_PLANS_REGISTERED", "GPU_started": False,
                      "GPU_present": gpu.returncode == 0, "fresh_cases": ["B3", "B4"],
                      "formal_episodes_per_case": 300, "total_new_formal_episodes": 600}))


if __name__ == "__main__":
    main()
