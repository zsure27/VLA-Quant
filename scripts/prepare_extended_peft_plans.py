"""Freeze separately recoverable B2/B3 server plans; no execution or SSH here."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qvla.extended_peft import sha, registry, VERSION


def build_plan(model_id, materials, session, output, python, expected_hostname):
    spec = json.loads((ROOT / "configs/experiments/b2_b3_v2_proprio_20261008.json").read_text())
    directory = session / model_id
    student = Path(materials["student80"]) if model_id == "B2" else directory / "student-state80"
    m = dict(materials, student80=str(student))
    material_file = output / f"{model_id}-materials.json"
    material_file.write_text(json.dumps(m, indent=2) + "\n")
    scripts = ("train_extended_peft.py", "run_locked_peft_stage.py", "prepare_extended_peft_plans.py",
               "prepare_a4_student_state80.py", "audit_extended_peft_eval.py", "audit_peft_calibration80.py", "vla_stage_supervisor.py")
    locked = {str(p): sha(p) for p in [*(ROOT / "scripts" / n for n in scripts),
        *sorted((ROOT / "qvla").glob("*.py")), *(ROOT / "diagnostics" / n for n in
        ("probe.py", "low_rank_recovery.py", "awq_interventions.py")),
        ROOT / "configs/model_registry_v1.json", ROOT / "configs/qvla-connected-422.txt",
        ROOT / "configs/experiments/b2_b3_v2_proprio_20261008.json", ROOT / "configs/backbones/awq_w2a16_all_eligible_spatial_v1.json",
        material_file, Path(materials["trajectory_split"]), Path(materials["calibration80"]) / "manifest.json",
        Path(materials["checkpoint"]) / "config.json", Path(materials["w2_g128"]), Path(materials["w2_g64"]),
        Path(materials["w4"]), Path(materials["b1_adapter"]), Path(materials["student80"]) / "manifest.json",
        *sorted(Path(materials["calibration80"]).glob("sample-*.npz")),
        *sorted(Path(materials["student80"]).glob("sample-*.npz"))]}
    lock_file = output / f"{model_id}-lock.json"
    lock_file.write_text(json.dumps({"version": VERSION, "expected_hostname": expected_hostname,
        "files": locked, "max_stage_seconds": spec["max_stage_seconds"],
        "max_plan_seconds": spec["max_plan_seconds"]}, indent=2) + "\n")
    env = {"PYTHONPATH": ":".join((str(ROOT / "diagnostics"), str(ROOT), materials["oft_root"], materials["libero_root"])),
           "CUDA_VISIBLE_DEVICES": "0", "PYTHONHASHSEED": "7", "PYTHONOPTIMIZE": "0", "WANDB_MODE": "disabled",
           "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": "0", "TOKENIZERS_PARALLELISM": "false",
           "TF_NUM_INTEROP_THREADS": "2", "TF_NUM_INTRAOP_THREADS": "4"}
    stages = []
    def stage(name, command, record=None):
        stages.append({"id": name, "cwd": str(ROOT), "env": env,
            "command": [python, str(ROOT / "scripts/run_locked_peft_stage.py"), "--lock", str(lock_file),
                        "--output", str(record or directory / "stage-records" / name), "--", *map(str, command)]})
    def evaluate(case, phase, offset, count):
        out = directory / f"eval-{phase}" / case
        base = {"B1": "A3", "B2": "A3", "B3": "A4"}.get(case, case)
        cmd = [python, ROOT / "qvla/run_eval_official_quant.py", "--model-id", case,
            "--method", "awq", "--weight-bits", "4" if base in ("BF16", "A0") else "2", "--activation-bits", "16",
            "--pretrained_checkpoint", materials["checkpoint"], "--profile", materials["w4"] if base in ("BF16", "A0") else materials["w2_g128"],
            "--official-root", materials["official_root"], "--libero_root", materials["libero_root"],
            "--task_suite_name", "libero_spatial", "--initial-state-offset", str(offset), "--num_trials_per_task", str(count),
            "--seed", "0", "--env-seed", "1", "--seed-protocol", "paired", "--trace-actions", "--trace-observations",
            "--awq-scope", "none" if base == "BF16" else "all", "--awq-candidate",
            "profile" if base in ("BF16", "A0") else "w2-attention-primary-g64-stage-w4" if base == "A3" else "w2-attention-no-clip-primary-g64",
            "--local_log_dir", out]
        if base not in ("BF16", "A0"):
            cmd += ["--awq-primary-group64-profile", materials["w2_g64"]]
        if base == "A3":
            cmd += ["--awq-w4-profile", materials["w4"], "--awq-w4-layers", "8,9,10,11,12,13,14,15,20,21,22,23"]
        if case in ("B1", "B2"):
            cmd += ["--awq-recovery-lora-state", materials["b1_adapter"], "--recovery-training-manifest", Path(materials["student80"]) / "manifest.json"]
        if case == "B2": cmd += ["--awq-visual-lora-state", directory / "train1000/adapter.pt", "--extended-peft-manifest", directory / "train1000/artifact.json"]
        if case == "B3": cmd += ["--awq-recovery-lora-state", directory / "train1000/adapter.pt", "--recovery-training-manifest", student / "manifest.json",
                                  "--extended-peft-manifest", directory / "train1000/artifact.json"]
        stage(f"{phase}-{case}", cmd, out)
    if model_id == "B3":
        evaluate("A4", "training-capture", 0, 4)
        stage("prepare-A4-student80", [python, ROOT / "scripts/prepare_a4_student_state80.py", "--run",
                                      directory / "eval-training-capture/A4", "--output", student])
    for steps in (10, 1000):
        name = "smoke10" if steps == 10 else "train1000"
        cmd = [python, ROOT / "scripts/train_extended_peft.py", "--model-id", model_id, "--materials", material_file,
               "--steps", str(steps), "--output", directory / name]
        if steps == 1000: cmd += ["--smoke-dir", directory / "smoke10"]
        stage(name, cmd)
    cases = ("B1", "B2", "A3", "BF16", "A0") if model_id == "B2" else ("A4", "B3", "B1", "BF16", "A0")
    for phase, count in (("micro", 1), ("first50", 5)):
        for case in cases: evaluate(case, phase, 20, count)
        stage(f"audit-{phase}", [python, ROOT / "scripts/audit_extended_peft_eval.py", "--root", directory,
              "--model-id", model_id, "--phase", phase, "--output", directory / f"{phase}-protocol-gate.json"])
    plan = {"schema_version": "1.0", "plan_id": f"{session.name}-{model_id}", "control_root": str(output / "control"),
            "terminal_state": "AWAITING_GATE_REVIEW", "stages": stages,
            "question": spec[model_id]["primary_difference"], "remaining_long_evaluation": "LOCKED"}
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--materials", required=True, type=Path)
    p.add_argument("--session", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--python", required=True)
    p.add_argument("--expected-hostname", required=True)
    a = p.parse_args()
    if socket.gethostname() != a.expected_hostname: raise ValueError("Actual instance identity differs")
    from scripts.train_extended_peft import materials_contract
    materials = json.loads(a.materials.read_text())
    materials_contract(materials, "B2")  # Validate inherited inputs before either plan is registered.
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if active: raise ValueError("GPU has existing compute processes; recover active plan before new registration")
    for name in ("backup_active_diagnostics.py", "vla_shutdown_remote.py"):
        if sha(Path(materials["closure_tools"]) / name) != sha(ROOT / "scripts" / name):
            raise ValueError("Sync and verify native closure tools before any GPU stage")
    for key in ("oft_root", "libero_root", "official_root"):
        if not Path(materials[key]).is_dir(): raise ValueError(f"Missing runtime: {key}")
    if a.output.exists(): raise ValueError("Plans already exist; recover same immutable plans")
    a.output.mkdir(parents=True)
    from scripts.vla_stage_supervisor import validate_plan
    for model_id in ("B2", "B3"):
        plan = build_plan(model_id, materials, a.session, a.output, a.python, a.expected_hostname)
        validate_plan(plan)
        (a.output / f"{model_id}-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"gate": "PASS_MATERIALS_PLANS_REGISTERED", "launch": "B2 first; boundary analysis before B3",
                      "output": str(a.output), "GPU_experiment_started": False}))


if __name__ == "__main__": main()
