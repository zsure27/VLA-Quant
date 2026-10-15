"""Register the next fixed Z/S 100-episode slice; no method/weight changes."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import socket
import sys

PHASES = ((10, 19), (20, 29), (30, 39), (40, 49))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_flag(command, flag, value):
    i = command.index(flag) + 1
    command[i] = str(value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--offset", type=int, choices=(10, 20, 30, 40), required=True)
    p.add_argument("--preregistration-commit", required=True)
    a = p.parse_args()
    c = a.control
    if socket.gethostname() != "autodl-container-90824aab47-6dc00ae3":
        raise ValueError("Wrong current 046")
    source = c / "source-v2"
    session = Path("/root/autodl-tmp/qvla-repro/backups/experiments/p2-shared-peft/20261015-046-quant-finetuning-v1")
    original_path = c / "B3-plans-v1/first50-plan.json"
    if sha(original_path) != "f6d2bf0a19cd6b6054f9ee6c4f77e18a07c99802624971bd69974c6ef5be8992":
        raise ValueError("Original immutable first50 plan changed")
    original = json.loads(original_path.read_text())
    phase = f"reset-{a.offset}-{a.offset + 9}"
    previous = "first50" if a.offset == 10 else f"reset-{a.offset - 10}-{a.offset - 1}"
    previous_offset, previous_count = (5, 5) if a.offset == 10 else (a.offset - 10, 10)
    gate_path = session / f"{previous}-gate.json"
    gate = json.loads(gate_path.read_text())
    expected = {(t, r) for t in range(10) for r in range(previous_offset, previous_offset + previous_count)}
    actual = {(r["task_id"], r["init_state_index"]) for r in gate["paired_rows"]}
    if (gate["gate"] != "PASS_B3_INITIALIZATION_PAIRED_PROTOCOL" or gate["paired_episodes"] != len(expected)
            or actual != expected or len(gate["paired_rows"]) != len(expected)):
        raise ValueError("Prior phase protocol/coverage gate incomplete")
    if (c / "STOP_NEW_GPU").exists():
        raise ValueError("Client stop-new-GPU marker present")
    # This continues already-admitted candidates. Initial >=12GiB was verified;
    # each slice reserves 6GiB plus a conservative 2GiB for this slice/archive.
    if shutil.disk_usage(session).free < 8 * 1024**3:
        raise ValueError("Continuation requires 6GiB reserve plus 2GiB slice/archive headroom")
    root = c / "B3-continuation-v1"
    root.mkdir(exist_ok=True)
    master = root / "PREREGISTERED_PHASES.json"
    if not master.exists():
        master.write_text(json.dumps({"version": "b3-initialization-continuation-v1-20261015",
            "phases": [{"offset": x, "end": y, "episodes_each": 100} for x, y in PHASES],
            "source_commit": a.preregistration_commit, "formal_total_each_including_first50": 450,
            "score_direction_not_a_gate": True}, indent=2) + "\n")
    output = root / phase
    if output.exists():
        raise ValueError("Continuation registration exists; inspect before any recovery")
    output.mkdir()
    lock = json.loads((c / "B3-plans-v1/runtime-lock.json").read_text())
    lock["files"][str(gate_path)] = sha(gate_path)
    lock["files"][str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    lock_path = output / "runtime-lock.json"
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    stages = []
    for template in original["stages"][-3:]:
        s = copy.deepcopy(template)
        replace_flag(s["command"], "--lock", lock_path)
        if template["id"].startswith("first50-"):
            variant = template["id"][len("first50-"):]
            destination = session / phase / variant
            s["id"] = phase + "-" + variant
            replace_flag(s["command"], "--output", destination)
            replace_flag(s["command"], "--initial-state-offset", a.offset)
            replace_flag(s["command"], "--num_trials_per_task", 10)
            replace_flag(s["command"], "--local_log_dir", destination)
            allowed = {"--lock", "--output", "--initial-state-offset", "--num_trials_per_task", "--local_log_dir"}
            old, new = template["command"], s["command"]
            changed = {i for i, (x, y) in enumerate(zip(old, new)) if x != y}
            permitted = {old.index(flag) + 1 for flag in allowed}
            if len(old) != len(new) or not changed <= permitted:
                raise ValueError("A GPU command field changed outside slice/output/lock")
        else:
            s["id"] = phase + "-audit"
            # Wrapper and child both use --output; address them separately.
            cmd = s["command"]
            cmd[cmd.index("--output") + 1] = str(session / "stage-records" / s["id"])
            inner = cmd.index("--") + 1
            tail = cmd[inner:]
            replace_flag(tail, "--root", session / phase)
            replace_flag(tail, "--offset", a.offset)
            replace_flag(tail, "--count", 10)
            replace_flag(tail, "--output", session / f"{phase}-gate.json")
            cmd[inner:] = tail
        stages.append(s)
    plan = {"schema_version": "1.0", "plan_id": session.name + "-" + phase,
            "control_root": str(output / "control"), "terminal_state": "AWAITING_GATE_REVIEW",
            "stages": stages, "preregistration_commit": a.preregistration_commit,
            "prior_gate": str(gate_path), "prior_gate_sha256": sha(gate_path),
            "question": "Continue the same fixed Z/S candidates, 100 each, protocol/resource decision only",
            "next_plan": "Analyze paired outcomes/failures/cost and re-read live quota/disk before next preregistered slice"}
    sys.path.insert(0, str(source))
    from scripts.vla_stage_supervisor import validate_plan
    validate_plan(plan)
    path = output / "plan.json"
    path.write_text(json.dumps(plan, indent=2) + "\n")
    receipt = {"gate": "PASS_B3_INITIALIZATION_CONTINUATION_COMMANDS_UNCHANGED", "phase": phase,
               "plan": str(path), "plan_sha256": sha(path), "GPU_started": False,
               "unique_wrapper_outputs": [s["command"][s["command"].index("--output") + 1] for s in stages]}
    if len(set(receipt["unique_wrapper_outputs"])) != 3:
        raise ValueError("Duplicate continuation wrapper outputs")
    (output / "registration.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
