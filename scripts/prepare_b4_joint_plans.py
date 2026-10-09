"""Register three immutable B4 plans on a verified user-opened instance; no launch."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qvla.b4_joint_peft import sha, SPEC, VERSION
from scripts.audit_b4_joint_eval import CASES, PHASES


def build_plans(materials, session, output, python, hostname):
    spec = json.loads(SPEC.read_text())
    material_file = output / "B4-materials.json"
    material_file.write_text(json.dumps(materials, indent=2) + "\n")
    code = [*sorted((ROOT / "qvla").glob("*.py")), *(ROOT / "scripts" / n for n in (
        "train_b4_joint_peft.py", "train_extended_peft.py", "prepare_b4_joint_plans.py",
        "audit_b4_joint_eval.py", "audit_extended_peft_eval.py", "audit_peft_calibration80.py",
        "run_locked_peft_stage.py", "vla_stage_supervisor.py")),
        *(ROOT / "diagnostics" / n for n in ("probe.py", "low_rank_recovery.py", "awq_interventions.py")),
        ROOT / "configs/model_registry_v1.json", ROOT / "configs/model_registry_b4_v1.json",
        ROOT / "configs/qvla-connected-422.txt", SPEC,
        ROOT / "configs/experiments/b2_b3_v3_proprio_20261008.json",
        ROOT / "configs/backbones/awq_w2a16_all_eligible_spatial_v1.json"]
    files = [*code, material_file, *(Path(materials[k]) for k in (
        "trajectory_split", "w2_g128", "w2_g64", "w4", "prior_closure_receipt")),
        Path(materials["checkpoint"]) / "config.json", Path(materials["checkpoint"]) / "dataset_statistics.json",
        *(Path(materials["b3_train_dir"]) / n for n in (
            "artifact.json", "adapter.pt", "initial-adapter.pt", "teacher-targets.pt", "teacher-input-hashes.json")),
        *(Path(materials[k]) / "manifest.json" for k in ("student80", "calibration80")),
        *sorted(Path(materials["student80"]).glob("sample-*.npz")),
        *sorted(Path(materials["calibration80"]).glob("sample-*.npz"))]
    if "closure_tools" in materials:
        files.extend(Path(materials["closure_tools"]) / name for name in (
            "backup_active_diagnostics.py", "vla_shutdown_remote.py"))
    fingerprints = {str(f): sha(f) for f in files}
    env = {"PYTHONPATH": ":".join((str(ROOT / "diagnostics"), str(ROOT), materials["oft_root"], materials["libero_root"])),
           "CUDA_VISIBLE_DEVICES": "0", "PYTHONHASHSEED": "7", "PYTHONOPTIMIZE": "0", "WANDB_MODE": "disabled",
           "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": "0", "TOKENIZERS_PARALLELISM": "false",
           "TF_NUM_INTEROP_THREADS": "2", "TF_NUM_INTRAOP_THREADS": "4"}
    plans = {}
    for phase in ("first50", "next50", "last100"):
        # One shared started-file bounds all three plans, including gate gaps.
        lock_file = output / "B4-runtime-lock.json"
        lock_file.write_text(json.dumps({"version": VERSION, "expected_hostname": hostname,
            "files": fingerprints, "max_stage_seconds": spec["max_stage_seconds"],
            "max_plan_seconds": spec["max_plan_seconds"]}, indent=2) + "\n")
        stages = []
        def stage(name, argv, directory=None):
            stages.append({"id": name, "cwd": str(ROOT), "env": env,
                "command": [python, str(ROOT / "scripts/run_locked_peft_stage.py"), "--lock", str(lock_file),
                            "--output", str(directory or session / "stage-records" / name), "--", *map(str, argv)]})
        def evaluate(case, part):
            offset, count = PHASES[part]
            reference = case in ("BF16", "A0")
            out = session / f"eval-{part}" / case
            cmd = [python, ROOT / ("qvla/run_eval_b4_joint.py" if case == "B4" else "qvla/run_eval_official_quant.py"),
                "--model-id", case, "--method", "awq", "--weight-bits", "4" if reference else "2", "--activation-bits", "16",
                "--pretrained_checkpoint", materials["checkpoint"], "--profile", materials["w4" if reference else "w2_g128"],
                "--official-root", materials["official_root"], "--libero_root", materials["libero_root"],
                "--task_suite_name", "libero_spatial", "--initial-state-offset", offset, "--num_trials_per_task", count,
                "--seed", 0, "--env-seed", 1, "--seed-protocol", "paired", "--trace-actions", "--trace-observations",
                "--awq-scope", "none" if case == "BF16" else "all", "--awq-candidate",
                "profile" if reference else "w2-attention-no-clip-primary-g64", "--local_log_dir", out]
            if not reference: cmd += ["--awq-primary-group64-profile", materials["w2_g64"]]
            if case in ("B3", "B4"):
                trained = Path(materials["b3_train_dir"]) if case == "B3" else session / "train1000"
                cmd += ["--awq-recovery-lora-state", trained / "adapter.pt", "--extended-peft-manifest", trained / "artifact.json",
                        "--recovery-training-manifest", Path(materials["student80"]) / "manifest.json"]
            stage(f"{part}-{case}", cmd, out)
        def audit(part):
            stage(f"audit-{part}", [python, ROOT / "scripts/audit_b4_joint_eval.py", "--root", session,
                "--phase", part, "--output", session / f"{part}-protocol-gate.json"])
        if phase == "first50":
            for steps in (10, 1000):
                name = "smoke10" if steps == 10 else "train1000"
                cmd = [python, ROOT / "scripts/train_b4_joint_peft.py", "--model-id", "B4",
                       "--materials", material_file, "--steps", steps, "--output", session / name]
                if steps == 1000: cmd += ["--smoke-dir", session / "smoke10"]
                stage(name, cmd)
            for case in CASES: evaluate(case, "micro")
            audit("micro")
        else:
            previous = "first50" if phase == "next50" else "next50"
            stage(f"verify-{previous}", [python, ROOT / "scripts/audit_b4_joint_eval.py",
                "--root", session, "--require-prior", previous])
        for case in CASES: evaluate(case, phase)
        audit(phase)
        if phase == "last100": audit("combined")
        plans[phase] = {"schema_version": "1.0", "plan_id": f"{session.name}-B4-{phase}",
            "control_root": str(output / "control"), "terminal_state": "AWAITING_GATE_REVIEW",
            "stages": stages, "question": "B4 minus B3 on fixed development reset20-39",
            "next_plan": "analyze protocol/results and current quota before starting the next registered plan"}
    return plans


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--materials", type=Path, required=True)
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--expected-hostname", required=True)
    a = p.parse_args()
    if socket.gethostname() != a.expected_hostname: raise ValueError("Wrong current instance hostname")
    if a.output.exists() or a.session.exists(): raise ValueError("Recover existing plan; never register over old output")
    materials = json.loads(a.materials.read_text())
    from scripts.train_b4_joint_peft import materials_contract
    materials_contract(materials, "B4")
    for name in ("backup_active_diagnostics.py", "vla_shutdown_remote.py"):
        if sha(Path(materials["closure_tools"]) / name) != sha(ROOT / "scripts" / name):
            raise ValueError("Sync and verify current native closure helpers first")
    for key in ("oft_root", "libero_root", "official_root"):
        if not Path(materials[key]).is_dir(): raise ValueError(f"Missing runtime: {key}")
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise ValueError("GPU busy; recover active plan rather than starting another")
    if shutil.disk_usage(a.session.parent).free < 12 * 1024**3:
        raise ValueError("B4 requires >=12 GiB persistent free space including >=6 GiB archive reserve")
    a.output.mkdir(parents=True)
    from scripts.vla_stage_supervisor import validate_plan
    for phase, plan in build_plans(materials, a.session, a.output, a.python, a.expected_hostname).items():
        validate_plan(plan)
        (a.output / f"B4-{phase}-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"gate": "PASS_B4_PLANS_REGISTERED", "GPU_started": False, "output": str(a.output)}))


if __name__ == "__main__": main()
