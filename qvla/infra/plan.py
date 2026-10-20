from __future__ import annotations

from pathlib import Path
import sys
import subprocess
from .common import canonical, digest, read_json
from .config import require, validate


def expand(argv, context):
    require(isinstance(argv, list) and argv and all(isinstance(v, str) and v for v in argv), "Command must be argv, never shell text")
    return [v.format_map(context) for v in argv]


def compile_plan(config, repo, run_dir):
    validate(config)
    repo, run_dir = Path(repo).resolve(), Path(run_dir).resolve()
    context = {"repo": str(repo), "run_dir": str(run_dir), "python": sys.executable,
               "checkpoint": config["model"]["checkpoint"], "task_id": config["task"]["id"]}
    quant, tuning = config["quantization"], config["finetuning"]
    binding = config.get("binding")
    stages, inputs = [], []

    def stage(name, argv, extra=None):
        ctx = dict(context, **(extra or {}))
        stages.append({"id": name, "argv": expand(argv, ctx), "cwd": str(repo)})

    for records in config.get("data", {}).values():
        for row in records:
            inputs.append({"path": row["manifest"], "sha256": row["sha256"]})
    profiles = quant.get("profiles", [])
    for profile in profiles:
        inputs.append(profile)
    if quant["method"] != "none" and not binding:
        require(profiles, "Provide calibrated profiles, or explicit calibration_argv")
    if binding:
        for gate in binding["gate_files"]:
            inputs.append({**gate, "path": gate["path"].format_map(context)})
        if tuning["method"] != "none" and tuning.get("train", True):
            stage("train", binding["training_argv"])
        if binding.get("post_training_audit_argv"):
            stage("training-audit", binding["post_training_audit_argv"])
    for suite in config["evaluation"]["suites"]:
        name = suite["name"]
        ctx = {"suite": name, "offset": str(suite["reset_offset"]),
               "count": str(suite["resets_per_task"]), "output": str(run_dir / name)}
        if binding:
            argv = binding["evaluation_argv"]
        elif quant["method"] == "none":
            argv = ["{python}", config["model"]["oft_root"] + "/experiments/robot/libero/run_libero_eval.py",
                    "--pretrained_checkpoint", "{checkpoint}", "--task_suite_name", "{suite}",
                    "--num_trials_per_task", "{count}", "--initial_state_offset", "{offset}",
                    "--local_log_dir", "{output}", "--center_crop", "True",
                    "--seed", str(config["evaluation"].get("seed", 7)),
                    "--env_seed", str(config["evaluation"].get("env_seed", 0)), "--seed_protocol", "paired"]
        else:
            argv = ["{python}", "{repo}/qvla/run_eval_official_quant.py", "--method", quant["method"],
                    "--weight-bits", str(quant["weight_bits"]), "--activation-bits", str(quant["activation_bits"]),
                    "--pretrained_checkpoint", "{checkpoint}", "--official-root", quant["official_root"],
                    "--task_suite_name", "{suite}", "--num_trials_per_task", "{count}",
                    "--initial-state-offset", "{offset}", "--local_log_dir", "{output}",
                    "--seed", str(config["evaluation"].get("seed", 7)),
                    "--env-seed", str(config["evaluation"].get("env_seed", 0)),
                    "--seed-protocol", "paired", "--trace-actions"]
            for profile in profiles:
                argv += ["--profile", profile["path"]]
            # Exact scientific recipe options remain explicit and are hashed with the plan.
            extra = quant.get("recipe_argv", [])
            forbidden = {"--method", "--weight-bits", "--activation-bits", "--task_suite_name",
                         "--num_trials_per_task", "--initial-state-offset", "--local_log_dir", "--seed",
                         "--env-seed", "--seed-protocol", "--pretrained_checkpoint", "--profile"}
            require(not (set(extra) & forbidden), "recipe_argv cannot override identity/cohort/output")
            argv += extra
        stage("eval-" + name, argv, ctx)
    if binding and binding.get("evaluation_audit_argv"):
        stage("evaluation-audit", binding["evaluation_audit_argv"])
    # Pin source files that are actual command arguments, independent of old report paths.
    sources = {}
    for path in (repo / "qvla/infra").glob("*.py"):
        sources[str(path)] = digest(path)
    for item in stages:
        for value in item["argv"]:
            path = Path(value)
            if path.is_file() and path.suffix == ".py" and repo in path.resolve().parents:
                sources[str(path)] = digest(path)
    plan = {"schema_version": 1, "config_sha256": canonical(config), "config": config,
            "run_dir": str(run_dir), "repo": str(repo), "stages": stages,
            "inputs": inputs, "source_sha256": sources,
            "note": "CPU plan compilation is not a model/numerical gate"}
    plan["git_commit"] = subprocess.check_output(
        ["git", "-c", "safe.directory=" + str(repo), "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    plan["plan_sha256"] = canonical(plan)
    return plan


def verify_plan(plan):
    copy = dict(plan)
    claimed = copy.pop("plan_sha256")
    require(canonical(copy) == claimed and canonical(plan["config"]) == plan["config_sha256"], "Plan/config changed")
    validate(plan["config"])
    for path, expected in plan["source_sha256"].items():
        require(digest(path) == expected, "Command source changed: " + path)
    for record in plan["inputs"]:
        require(digest(record["path"]) == record["sha256"], "Input SHA changed: " + record["path"])
        if record.get("expected_gate"):
            require(read_json(record["path"]).get("gate") == record["expected_gate"], "Numerical/data gate rejected")
    # Same legacy full-content identity algorithm; does not import torch or load a model.
    from qvla.baseline_contract import checkpoint_identity
    require(checkpoint_identity(plan["config"]["model"]["checkpoint"])["sha256"] ==
            plan["config"]["model"]["identity_sha256"], "Checkpoint content mismatch")
    from .data import audit_manifests
    audit_manifests(plan["config"])
