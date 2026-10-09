"""B4 joint contracts; archived B2/B3 source bytes remain unchanged."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import re

from qvla.extended_peft import (ROOT, sha, canonical_sha, target_names as old_targets,
    validate_shapes as old_shapes, student_contract as old_student,
    zero_state, attach_state)

VERSION = "b4-joint-v1-20261009"
SPEC = ROOT / "configs/experiments/b4_joint_v1_20261009.json"
MODEL_IDS = ("B4",)
SOURCES = ("scripts/train_b4_joint_peft.py", "qvla/b4_joint_peft.py",
           "qvla/run_eval_b4_joint.py", "qvla/recovery_lora.py", "qvla/extended_peft.py",
           "qvla/model_registry.py", "qvla/run_eval_official_quant.py", "qvla/data_roles.py",
           "qvla/action_jacobian_batch.py", "qvla/extended_proprio.py", "diagnostics/probe.py",
           "diagnostics/low_rank_recovery.py", "diagnostics/awq_interventions.py")


def target_names(model_id="B4"):
    if model_id != "B4": raise ValueError("Joint contract only accepts B4")
    return old_targets("B3") | old_targets("B2")


def descriptors(model_id, modules, targets):
    from qvla.extended_peft import descriptors as old_descriptors
    target_names(model_id)
    return sorted(old_descriptors("B3", modules, targets) + old_descriptors("B2", modules, targets),
                  key=lambda r: r["name"])


def validate_shapes(shapes, model_id="B4", rows=None):
    names = target_names(model_id)
    expected = {f"{n}.awq_recovery_lora.{j}.weight" for n in names for j in (0, 1)}
    if set(shapes) != expected: raise ValueError("B4 requires 420 Linear targets / 840 matrices")
    if rows is not None and (len(rows) != 420 or {r["name"] for r in rows} != names):
        raise ValueError("B4 descriptor coverage mismatch")
    for kind in ("B3", "B2"):
        subset = old_targets(kind)
        old_shapes({k: v for k, v in shapes.items() if k.rsplit(".awq_recovery_lora.", 1)[0] in subset},
                   kind, [r for r in rows if r["name"] in subset] if rows is not None else None)


def load_state(path, model_id="B4", rows=None):
    import torch
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or any(not isinstance(v, torch.Tensor) or
            not v.is_floating_point() or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Joint adapter must contain finite floating tensors")
    validate_shapes({k: tuple(v.shape) for k, v in state.items()}, model_id, rows)
    return state


def trainable_contract(model, model_id="B4"):
    expected = {f"{n}.awq_recovery_lora.{j}.weight" for n in target_names(model_id) for j in (0, 1)}
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(named) != expected: raise ValueError("B4 trainable whitelist mismatch")
    return named


def student_contract(directory, model_id="B4"):
    target_names(model_id)
    if sha(Path(directory) / "manifest.json") != json.loads(SPEC.read_text())["B3_training_manifest_sha256"]:
        raise ValueError("B4 must reuse the exact B3 A4 student-state80")
    return old_student(directory, "B3")


def inherited_contract(materials):
    """Pin B3 pre-training initialization and labels, never its trained state."""
    spec = json.loads(SPEC.read_text())
    record_path = Path(materials["b3_train_dir"]) / "artifact.json"
    if sha(record_path) != spec["B3_training_artifact_sha256"]:
        raise ValueError("B3 source artifact mismatch")
    record = json.loads(record_path.read_text())
    if record["gate"] != "PASS_TRAIN_CONTRACT" or record["model_id"] != "B3":
        raise ValueError("Require the accepted B3 training artifact")
    for relative, expected in record["source_sha256"].items():
        if sha(ROOT / relative) != expected: raise ValueError(f"Archived B3 source changed: {relative}")
    if record["training_manifest_sha256"] != sha(Path(materials["student80"]) / "manifest.json"):
        raise ValueError("B3/B4 training data differ")
    for name, expected in spec["B3_reuse_sha256"].items():
        if record["reuse_files"][name] != expected or sha(Path(materials["b3_train_dir"]) / name) != expected:
            raise ValueError(f"B3 pre-training cache mismatch: {name}")
    return record


def closure_contract(materials):
    """Only a complete, SHA-bound prior closure receipt admits new training."""
    receipt = json.loads(Path(materials["prior_closure_receipt"]).read_text())
    if (receipt.get("gate") != "PASS_PRIOR_CLOSURE" or
            receipt.get("local_gate") != "PASS_LOCAL_ARCHIVE_SHA256" or
            receipt.get("server_gate") != "PASS_PERSISTENT_ARCHIVE" or
            receipt.get("github_repository") != "zsure27/VLA-Quant" or
            not re.fullmatch(r"[0-9a-f]{40}", receipt.get("verified_remote_sha", ""))):
        raise ValueError("Complete prior closure and GitHub verification before B4")
    archive = Path(materials["prior_server_archive"]).resolve(strict=True)
    for name, count in (("SHA256SUMS.txt", 13), ("RESULTS_SHA256SUMS.txt", 18494)):
        expected = receipt["server_manifest_sha256"][name]
        if expected != receipt["local_manifest_sha256"][name] or sha(archive / name) != expected:
            raise ValueError("Prior server/local archive manifests differ")
        lines = (archive / name).read_text().splitlines()
        if len(lines) != count or receipt["verified_counts"].get(name) != count:
            raise ValueError("Incomplete prior archive coverage")
        for line in lines:
            digest, relative = line.split(maxsplit=1)
            path = (archive / relative.lstrip("*")).resolve(strict=True)
            if archive not in path.parents or sha(path) != digest:
                raise ValueError(f"Prior server archive SHA/path mismatch: {relative}")
    return receipt


def mapped_args(args):
    other = copy.copy(args)
    if other.model_id != "B4": raise ValueError("B4 evaluator requires --model-id B4")
    other.model_id = "B3"  # identical A4 quantization recipe, distinct adapter contract below
    return other


def validate_request(args):
    from qvla.model_registry import validate_request as old_validate
    old_validate(mapped_args(args))


def validate_profile_provenance(args):
    from qvla.model_registry import validate_profile_provenance as old_validate
    old_validate(mapped_args(args))


def validate_effective_plan(model_id, targets):
    from qvla.model_registry import validate_effective_plan as old_validate
    target_names(model_id)
    old_validate("A4", targets)


def validate_artifact(args):
    record = json.loads(args.extended_peft_manifest.read_text())
    if (record.get("version") != VERSION or record.get("model_id") != "B4" or
            record.get("steps") != 1000 or record.get("gate") != "PASS_TRAIN_CONTRACT" or
            record.get("training_state_space") != "policy_normalized_proprio" or
            record.get("spec_sha256") != sha(SPEC)):
        raise ValueError("Require versioned B4 full-training artifact")
    if record.get("source_sha256") != {p: sha(ROOT / p) for p in SOURCES}:
        raise ValueError("B4 source provenance mismatch")
    spec = json.loads(SPEC.read_text())
    if (record.get("training_manifest_sha256") != spec["B3_training_manifest_sha256"] or
            args.recovery_training_manifest is None or
            sha(args.recovery_training_manifest) != record["training_manifest_sha256"] or
            record.get("inherited_B3_artifact_sha256") != spec["B3_training_artifact_sha256"] or
            record.get("inherited_reuse_sha256") != spec["B3_reuse_sha256"] or
            record.get("parameters") != 26710528):
        raise ValueError("B4 inherited training/init/teacher/parameter contract differs")
    if (record.get("zero_output_equal") is not True or record.get("reload_output_equal") is not True or
            record.get("visual_zero_matches_B3_initial") is not True or
            not record.get("smoke_artifact_sha256") or not record.get("prior_closure_receipt_sha256") or
            not record.get("frozen_before_sha256") or
            record["frozen_before_sha256"] != record.get("frozen_after_sha256") or
            record.get("target_sha256") != canonical_sha(record["targets"])):
        raise ValueError("B4 actual-model smoke/frozen/reload contracts incomplete")
    if args.awq_recovery_lora_state is None or sha(args.awq_recovery_lora_state) != record["adapter_sha256"]:
        raise ValueError("B4 adapter SHA mismatch")
    validate_shapes(record["tensor_shapes"], "B4", record["targets"])
    if (args.task_suite_name != "libero_spatial" or args.initial_state_offset < 20 or
            args.initial_state_offset + args.num_trials_per_task > 40):
        raise ValueError("B4 only admits the preregistered development reset20-39")
    return record
