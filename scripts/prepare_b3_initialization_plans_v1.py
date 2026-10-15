"""Register fixed-A4 Z/S smoke, training and at-most-first50; never launch."""
import argparse
import json
from pathlib import Path
import socket
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvla.b3_initialization_v1 import ROOT, SOURCES, SPEC, VERSION, inherited_cache_contract
from qvla.extended_peft import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("materials", "output", "session", "resume-gate", "observation-gate", "cpu-gate"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--expected-hostname", required=True)
    p.add_argument("--python", required=True)
    a = p.parse_args()
    if socket.gethostname() != a.expected_hostname or a.output.exists():
        raise ValueError("Wrong instance or registration already exists")
    for path, gate in ((a.resume_gate, "PASS_TRANSFERRED_ARCHIVE_AND_B3_INPUTS"),
                       (a.observation_gate, "PASS_CPU_PROTOCOL_OBSERVATION_BRIDGE"),
                       (a.cpu_gate, "PASS_B3_INITIALIZATION_CPU_CONTRACTS")):
        if json.loads(path.read_text())["gate"] != gate:
            raise ValueError(f"Missing prerequisite gate: {gate}")
    m = json.loads(a.materials.read_text())
    inherited_cache_contract(m)
    a.output.mkdir(parents=True)
    lock = a.output / "runtime-lock.json"
    files = {str(ROOT / name): sha(ROOT / name) for name in SOURCES}
    for path in (SPEC, a.materials, a.resume_gate, a.observation_gate, a.cpu_gate,
                 ROOT / "scripts/prepare_b3_initialization_plans_v1.py",
                 ROOT / "scripts/audit_b3_initialization_v1.py", ROOT / "scripts/run_locked_peft_stage.py",
                 Path(m["oft_root"]) / "experiments/robot/libero/run_libero_eval.py",
                 Path(m["oft_root"]) / "experiments/robot/openvla_utils.py",
                 *(Path(m[key]) for key in ("w2_g128", "w2_g64", "w4")),
                 Path(m["student80"]) / "manifest.json",
                 *(Path(m["shared_b3_smoke_dir"]) / name for name in ("artifact.json", "initial-adapter.pt", "teacher-targets.pt", "teacher-input-hashes.json"))):
        files[str(path)] = sha(path)
    lock.write_text(json.dumps({"version": VERSION, "expected_hostname": a.expected_hostname,
                               "files": files, "max_stage_seconds": 10800, "max_plan_seconds": 21600}, indent=2) + "\n")
    env = {"PYTHONPATH": ":".join((str(ROOT), str(ROOT / "diagnostics"), m["oft_root"], m["libero_root"])),
           "CUDA_VISIBLE_DEVICES": "0", "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": "0", "PYOPENGL_PLATFORM": "egl",
           "PYTHONHASHSEED": "7", "WANDB_MODE": "disabled", "TOKENIZERS_PARALLELISM": "false",
           "TF_NUM_INTEROP_THREADS": "1", "TF_NUM_INTRAOP_THREADS": "1"}
    stages = []
    def add(name, command, directory=None):
        stages.append({"id": name, "cwd": str(ROOT), "env": env,
                       "command": list(map(str, [a.python, ROOT / "scripts/run_locked_peft_stage.py", "--lock", lock,
                                       "--output", directory or a.session / "stage-records" / name, "--", *command]))})
    for steps in (10, 1000):
        for variant in ("standard-zero", "response-svd"):
            directory = a.session / variant / f"steps-{steps}"
            cmd = [a.python, ROOT / "scripts/train_b3_initialization_v1.py", "--model-id", "B3",
                   "--initialization", variant, "--materials", a.materials, "--steps", steps, "--output", directory]
            if steps == 1000:
                cmd += ["--smoke-dir", a.session / variant / "steps-10"]
            add(f"train-{steps}-{variant}", cmd)
        add(f"audit-training-{steps}", [a.python, ROOT / "scripts/audit_b3_initialization_v1.py", "--root", a.session,
                                        "--steps", steps, "--output", a.session / f"train-{steps}-pair-gate.json"])
    for phase, count in (("micro", 1), ("first50", 5)):
        for variant in ("standard-zero", "response-svd"):
            directory = a.session / phase / variant
            trained = a.session / variant / "steps-1000"
            cmd = [a.python, ROOT / "qvla/run_eval_b3_initialization_v1.py", "--model-id", "B3", "--b3-initialization", variant,
                   "--method", "awq", "--weight-bits", 2, "--activation-bits", 16, "--profile", m["w2_g128"],
                   "--awq-primary-group64-profile", m["w2_g64"], "--awq-candidate", "w2-attention-no-clip-primary-g64",
                   "--official-root", m["official_root"], "--pretrained_checkpoint", m["checkpoint"], "--libero_root", m["libero_root"],
                   "--task_suite_name", "libero_spatial", "--initial-state-offset", 5, "--num_trials_per_task", count,
                   "--seed", 0, "--env-seed", 1, "--seed-protocol", "paired", "--trace-actions", "--trace-observations",
                   "--awq-recovery-lora-state", trained / "adapter.pt", "--extended-peft-manifest", trained / "artifact.json",
                   "--recovery-training-manifest", Path(m["student80"]) / "manifest.json", "--local_log_dir", directory]
            add(f"{phase}-{variant}", cmd, directory)
        add(f"audit-{phase}", [a.python, ROOT / "scripts/audit_b3_initialization_v1.py", "--root", a.session / phase,
                               "--offset", 5, "--count", count, "--output", a.session / f"{phase}-gate.json"])
    plan = {"schema_version": "1.0", "plan_id": a.session.name + "-first50", "control_root": str(a.output / "control"),
            "terminal_state": "AWAITING_GATE_REVIEW", "stages": stages,
            "question": "Fixed A4, initialization-only Z/S; same training inputs/teacher/order/budget",
            "formal_registered_total_per_variant": 450, "this_plan_formal_per_variant": 50,
            "micro_overlap_excluded": True, "next_plan": "Analyze first50 protocol, failures/cost, live quota/disk; preregister continuation regardless of score direction"}
    from scripts.vla_stage_supervisor import validate_plan
    validate_plan(plan)
    path = a.output / "first50-plan.json"
    path.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"gate": "PASS_B3_INITIALIZATION_PLAN_REGISTERED", "plan": str(path), "stages": len(stages), "gpu_started": False}))


if __name__ == "__main__":
    main()
