"""Versioned B2/B3 contracts. No relaxation of archived B0/B1 loaders."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION = "b2-b3-v3-proprio-20261008"
ARTIFACT_SOURCES = ("scripts/train_extended_peft.py", "qvla/extended_peft.py", "qvla/recovery_lora.py",
    "qvla/model_registry.py", "qvla/run_eval_official_quant.py", "qvla/data_roles.py",
    "qvla/action_jacobian_batch.py", "qvla/extended_proprio.py", "diagnostics/probe.py", "diagnostics/low_rank_recovery.py",
    "diagnostics/awq_interventions.py")
FAMILIES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
VISION = re.compile(r"^vision_backbone\.(?:fused_)?featurizer\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:fc1|fc2))$")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1048576), b""):
            h.update(part)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def registry():
    return json.loads((ROOT / "configs/model_registry_v1.json").read_text(encoding="utf-8"))


def target_names(model_id):
    if model_id == "B2":
        names = [n for n in (ROOT / "configs/qvla-connected-422.txt").read_text().splitlines() if VISION.fullmatch(n)]
        if len(names) != 196:
            raise ValueError("B2 requires the frozen 196 transformer Linear targets")
        return set(names)
    if model_id == "B3":
        return {f"language_model.model.layers.{i}.{f}" for i in range(32) for f in FAMILIES}
    raise ValueError("Extended adapter must be B2 or B3")


def descriptors(model_id, modules, targets):
    import torch
    rows = []
    for name in sorted(target_names(model_id)):
        module = modules.get(name)
        if not isinstance(module, torch.nn.Linear) or name not in targets:
            raise ValueError(f"Unregistered/non-Linear adapter target: {name}")
        entry, bits, group = targets[name]
        if bits != 2 or tuple(entry["shape"]) != tuple(module.weight.shape):
            raise ValueError(f"Adapter must match a W2 target shape: {name}")
        if min(module.weight.shape) <= 8:
            raise ValueError("Rank8 must be smaller than target dimensions")
        rows.append({"name": name, "shape": list(module.weight.shape), "bits": bits, "group": group,
                     "rank": 8, "parameters": 8 * sum(module.weight.shape)})
    return rows


def validate_shapes(shapes, model_id, rows=None):
    names = target_names(model_id)
    expected = {f"{n}.awq_recovery_lora.{j}.weight" for n in names for j in (0, 1)}
    if set(shapes) != expected:
        raise ValueError(f"{model_id} requires exact adapter coverage ({len(expected)} matrices)")
    by_name = {r["name"]: r for r in rows} if rows is not None else None
    if by_name is not None and (len(rows) != len(names) or set(by_name) != names):
        raise ValueError("Target descriptor coverage mismatch")
    for name in names:
        down, up = [tuple(shapes[f"{name}.awq_recovery_lora.{j}.weight"]) for j in (0, 1)]
        if len(down) != 2 or len(up) != 2 or down[0] != 8 or up[1] != 8 or min(*down, *up) <= 0:
            raise ValueError(f"Invalid rank8 state: {name}")
        if by_name is not None:
            r = by_name[name]
            expected_group = 128 if ".fused_featurizer." in name else 64
            if r["bits"] != 2 or r["group"] != expected_group or r["rank"] != 8 or down != (8, r["shape"][1]) or up != (r["shape"][0], 8):
                raise ValueError(f"Descriptor/tensor mismatch: {name}")


def load_state(path, model_id, rows=None):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or any(not isinstance(v, torch.Tensor) or not v.is_floating_point()
                                           or not torch.isfinite(v).all() for v in payload.values()):
        raise ValueError("Adapter must contain finite floating tensors")
    validate_shapes({k: tuple(v.shape) for k, v in payload.items()}, model_id, rows)
    return payload


def zero_state(module, name):
    """Unscaled alpha/rank=1 branch; same deterministic name seed as B1."""
    import math
    import torch
    devices = [module.weight.device.index] if module.weight.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))
        a = torch.empty(8, module.in_features, device=module.weight.device, dtype=module.weight.dtype)
        torch.nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        b = torch.zeros(module.out_features, 8, device=module.weight.device, dtype=module.weight.dtype)
    return a, b


def attach_state(model, state, trainable_names=()):
    from qvla.recovery_lora import attach_recovery_lora_state
    modules = dict(model.named_modules())
    names = {k.rsplit(".awq_recovery_lora.", 1)[0] for k in state}
    trainable_names = set(trainable_names)
    if not trainable_names <= names:
        raise ValueError("Trainable names outside attached targets")
    for n in sorted(names):
        attach_recovery_lora_state(modules[n], state[f"{n}.awq_recovery_lora.0.weight"],
                                  state[f"{n}.awq_recovery_lora.1.weight"])
        modules[n].awq_recovery_lora.requires_grad_(n in trainable_names)


def trainable_contract(model, model_id):
    expected = {f"{n}.awq_recovery_lora.{j}.weight" for n in target_names(model_id) for j in (0, 1)}
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(named) != expected:
        raise ValueError("Trainable whitelist mismatch; frozen base/B1 must have no gradients")
    return named


def student_contract(directory, model_id):
    from qvla.data_roles import audit_training_inputs
    paths = sorted(Path(directory).glob("sample-*.npz"))
    audit = audit_training_inputs(paths, None)
    import numpy as np
    for path in paths:
        with np.load(path, allow_pickle=False) as sample:
            if sample["state"].shape != (8,) or not np.isfinite(sample["state"]).all():
                raise ValueError("Student input must contain finite 8D proprio")
            for key in ("image", "wrist_image"):
                value = sample[key]
                if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3 or min(value.shape) <= 0:
                    raise ValueError("Student input images must be uint8 HxWx3")
    manifest = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
    rows = manifest["samples"]
    keys = [(r["task_id"], r["init_state_index"], r["query_in_episode"]) for r in rows]
    expected = {(t, r, q) for t in range(10) for r in range(4) for q in (0, 1)}
    if len(keys) != 80 or set(keys) != expected or manifest["role"] != "student_state_train":
        raise ValueError("Require 40 training episodes reset0-3, exactly query0-1 (80 observations)")
    if model_id == "B2":
        if sha(Path(directory) / "manifest.json") != registry()["models"]["B1"]["artifact"]["training_manifest_sha256"]:
            raise ValueError("B2 must reuse the exact archived B1 student-state80")
        # The archived manifest predates state-space labels. Its pinned SHA and
        # on-policy capture provenance identify the stored state as the policy's
        # already normalized proprio, not the raw pre-get_action trace.
    elif model_id == "B3":
        if manifest.get("source_model_id") != "A4" or not manifest.get("source_run_sha256"):
            raise ValueError("B3 requires freshly captured, hash-registered A4 student states")
        if manifest.get("state_space") != "policy_normalized_proprio":
            raise ValueError("B3 capture must explicitly identify policy-normalized proprio")
    else:
        raise ValueError("Unknown training candidate")
    return paths, audit


def validate_artifact(args):
    path = getattr(args, "extended_peft_manifest", None)
    if path is None:
        raise ValueError("B2/B3 require --extended-peft-manifest")
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record.get("version") != VERSION or record.get("model_id") != args.model_id or
            record.get("steps") != 1000 or record.get("gate") != "PASS_TRAIN_CONTRACT" or
            record.get("training_state_space") != "policy_normalized_proprio"):
        raise ValueError("Require the versioned full-training artifact, not a smoke checkpoint")
    spec_path = ROOT / "configs/experiments/b2_b3_v3_proprio_20261008.json"
    if record.get("spec_sha256") != sha(spec_path):
        raise ValueError("Experiment specification changed")
    if set(record["source_sha256"]) != set(ARTIFACT_SOURCES):
        raise ValueError("Incomplete artifact source provenance")
    if (record.get("zero_output_equal") is not True or record.get("reload_output_equal") is not True
            or not record.get("smoke_artifact_sha256") or not record.get("frozen_before_sha256")
            or record["frozen_before_sha256"] != record.get("frozen_after_sha256")
            or record.get("target_sha256") != canonical_sha(record["targets"])):
        raise ValueError("Smoke/frozen/reload/target contract is incomplete")
    for relative, expected in record["source_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Artifact implementation changed: {relative}")
    training = args.recovery_training_manifest
    if training is None or sha(training) != record["training_manifest_sha256"]:
        raise ValueError("Training provenance SHA mismatch")
    rows = json.loads(training.read_text(encoding="utf-8"))["samples"]
    trained = {(r["task_id"], r["init_state_index"]) for r in rows}
    evaluated = {(t, r) for t in range(10) for r in range(args.initial_state_offset,
                 args.initial_state_offset + args.num_trials_per_task)}
    if args.task_suite_name != "libero_spatial" or trained & evaluated:
        raise ValueError("Training/evaluation reset overlap or unregistered suite")
    state = args.awq_visual_lora_state if args.model_id == "B2" else args.awq_recovery_lora_state
    if state is None or sha(state) != record["adapter_sha256"]:
        raise ValueError("Candidate adapter SHA mismatch")
    if args.model_id == "B2" and sha(args.awq_recovery_lora_state) != registry()["models"]["B1"]["artifact"]["adapter_sha256"]:
        raise ValueError("B2 frozen B1 mismatch")
    validate_shapes(record["tensor_shapes"], args.model_id, record["targets"])
    return record
