from __future__ import annotations

from datetime import datetime
import math
import re
from .registry import SUITES, QUANTIZERS, STRATEGIES


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(config):
    require(config.get("schema_version") == 1, "schema_version must be 1")
    task = config["task"]
    require(re.fullmatch(r"[a-zA-Z0-9_-]+", task["id"]), "Invalid task.id")
    require(re.search(r"[\u4e00-\u9fff]", task["name_zh"]) and
            not re.search(r'[<>:"/\\|?*\r\n]', task["name_zh"]), "Use a Chinese task name safe for filenames")
    start = datetime.fromisoformat(task["started_at"].replace("Z", "+00:00"))
    require(start.tzinfo is not None, "started_at needs an explicit timezone")
    require(task.get("hypothesis") and task.get("changed_variable"), "State hypothesis and changed variable")
    require(config.get("model", {}).get("checkpoint"), "Explicit model.checkpoint required")
    require(config["model"].get("identity_sha256") and
            re.fullmatch(r"[a-f0-9]{64}", config["model"]["identity_sha256"]), "Pin model identity SHA256")
    quant = config["quantization"]
    require(quant["method"] in QUANTIZERS, "Register quantizer before use")
    spec = QUANTIZERS[quant["method"]]
    require(quant["weight_bits"] in spec["weight_bits"] and
            quant["activation_bits"] in spec["activation_bits"], "Unsupported W/A combination")
    strategy = config["finetuning"]["method"]
    require(strategy in STRATEGIES, "Register strategy before use")
    suite_names = [s["name"] for s in config["evaluation"]["suites"]]
    require(suite_names and len(suite_names) == len(set(suite_names)), "Select unique evaluation suites")
    eval_keys = set()
    for suite in config["evaluation"]["suites"]:
        require(suite["name"] in SUITES, "Unknown suite")
        require(suite.get("role") == "development", "This exploration framework does not read holdout labels")
        offset, count = suite["reset_offset"], suite["resets_per_task"]
        require(type(offset) is int and type(count) is int and offset >= 0 and count > 0, "Invalid reset range")
        # Existing generic evaluator executes all tasks. A plugin can add subsets later.
        tasks = suite.get("task_ids", list(range(SUITES[suite["name"]]["tasks"])))
        require(tasks == list(range(SUITES[suite["name"]]["tasks"])), "Built-in evaluator requires all suite tasks")
        eval_keys |= {(suite["name"], t, k) for t in tasks for k in range(offset, offset + count)}
    for role in ("training", "calibration"):
        for row in config.get("data", {}).get(role, []):
            require(row.get("role") == role, "Explicit data role required")
            require(row["suite"] in SUITES, "Unknown data suite")
            require(row.get("namespace") in ("simulator_reset", "offline_trajectory"), "Explicit data namespace required")
            require(row.get("manifest") and re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")), "Pin input manifest SHA")
            if row["namespace"] == "simulator_reset":
                require(row.get("task_ids") and row.get("resets"), "Enumerate training/calibration reset identities")
                keys = {(row["suite"], t, k) for t in row["task_ids"] for k in row["resets"]}
                require(not (keys & eval_keys), "Training/calibration overlaps evaluation resets")
    binding = config.get("binding")
    if binding or STRATEGIES[strategy]["binding_required"] or spec["backend"] == "command":
        require(binding and binding.get("id") and binding.get("gate_files"), "Versioned command binding and gates required")
        require(set(suite_names) <= set(binding.get("verified_suites", [])), "Binding does not support selected suites")
        require(quant["method"] in binding.get("quantizers", []), "Binding does not support quantizer")
        require(strategy in binding.get("strategies", []), "Binding does not support strategy")
        require(binding.get("evaluation_argv"), "Binding requires an evaluation command")
        if strategy != "none" and config["finetuning"].get("train", True):
            require(binding.get("training_argv"), "Training command required")
    for gate in (binding or {}).get("gate_files", []):
        require(gate.get("path") and gate.get("expected_gate") and
                re.fullmatch(r"[a-f0-9]{64}", gate.get("sha256", "")), "Gate path/SHA/expected value required")
    control = config["control"]
    require(type(control.get("heartbeat_seconds")) is int and control["heartbeat_seconds"] >= 60, "Heartbeat minimum 60s; default 900")
    stop, close = control["stop_expansion_remaining_percent"], control["close_remaining_percent"]
    require(all(type(x) in (int, float) and math.isfinite(x) for x in (stop, close)) and
            3 < close <= stop < 100, "Require 3 < close <= stop < 100")
    for name in ("git_sync_argv", "backup_argv"):
        require(control.get(name) and all(isinstance(x, str) and x for x in control[name]), f"{name} required for closure")
    require(control.get("shutdown_provider") == "autodl_native", "Select an implemented shutdown provider")
    return config
