"""Fixed-A4 B3 initialization comparison; archived validators stay unchanged."""
from __future__ import annotations
import json
from pathlib import Path
from qvla.extended_peft import (ROOT, sha, canonical_sha, load_state, validate_shapes,
    target_names, descriptors, student_contract, zero_state, attach_state, trainable_contract)

VERSION = "b3-initialization-v1-20261015"
SPEC = ROOT / "configs/experiments/b3_initialization_v1_20261015.json"
INITIALIZATIONS = ("standard-zero", "response-svd")
from scripts.train_extended_peft import SOURCES as LEGACY_SOURCES
SOURCES = tuple(LEGACY_SOURCES) + ("qvla/b3_initialization_v1.py",
    "scripts/train_b3_initialization_v1.py", "qvla/run_eval_b3_initialization_v1.py")


def inherited_cache_contract(materials):
    cache = Path(materials["shared_b3_smoke_dir"])
    prior = json.loads((cache / "artifact.json").read_text())
    if (prior.get("version") != "b2-b3-v3-proprio-20261008" or prior.get("model_id") != "B3"
            or prior.get("steps") != 10 or prior.get("gate") != "PASS_SMOKE10"
            or prior.get("training_state_space") != "policy_normalized_proprio"
            or prior.get("spec_sha256") != sha(ROOT / "configs/experiments/b2_b3_v3_proprio_20261008.json")
            or prior.get("training_manifest_sha256") != sha(Path(materials["student80"]) / "manifest.json")):
        raise ValueError("Historical initial factors/cache have the wrong candidate/data/spec")
    if prior.get("source_sha256") != {p: sha(ROOT / p) for p in LEGACY_SOURCES}:
        raise ValueError("Historical SVD/cache implementation identity changed")
    expected = json.loads(SPEC.read_text())["inherited_cache_sha256"]
    if prior.get("reuse_files") != expected:
        raise ValueError("SVD initial factors and teacher cache are not the preregistered bytes")
    for name, digest in expected.items():
        if sha(cache / name) != digest:
            raise ValueError(f"Historical cache SHA mismatch: {name}")
    if (prior.get("zero_output_equal") is not True or prior.get("reload_output_equal") is not True
            or prior.get("frozen_before_sha256") != prior.get("frozen_after_sha256")):
        raise ValueError("Inherited smoke numerical contracts incomplete")
    return prior


def validate_artifact(args):
    path = getattr(args, "extended_peft_manifest", None)
    if path is None or args.model_id != "B3":
        raise ValueError("New initialization evaluator requires B3 and its new artifact")
    record = json.loads(path.read_text())
    if (record.get("version") != VERSION or record.get("model_id") != "B3"
            or record.get("initialization") != args.b3_initialization
            or record.get("initialization") not in INITIALIZATIONS
            or record.get("steps") != 1000 or record.get("gate") != "PASS_TRAIN_CONTRACT"
            or record.get("training_state_space") != "policy_normalized_proprio"
            or record.get("spec_sha256") != sha(SPEC)):
        raise ValueError("Wrong initialization candidate/full-training artifact/spec")
    if record.get("source_sha256") != {p: sha(ROOT / p) for p in SOURCES}:
        raise ValueError("Incomplete or changed initialization implementation")
    if (record.get("zero_output_equal") is not True or record.get("reload_output_equal") is not True
            or not record.get("smoke_artifact_sha256") or not record.get("frozen_before_sha256")
            or record["frozen_before_sha256"] != record.get("frozen_after_sha256")
            or record.get("target_sha256") != canonical_sha(record["targets"])
            or record.get("parameters") != 19988480):
        raise ValueError("Missing gradient/frozen/reload/parameter contract")
    training = args.recovery_training_manifest
    if training is None or sha(training) != record["training_manifest_sha256"]:
        raise ValueError("Training provenance changed")
    rows = json.loads(training.read_text())["samples"]
    trained = {(r["task_id"], r["init_state_index"]) for r in rows}
    evaluated = {(t, r) for t in range(10) for r in range(args.initial_state_offset,
                 args.initial_state_offset + args.num_trials_per_task)}
    if args.task_suite_name != "libero_spatial" or trained & evaluated or args.initial_state_offset < 5:
        raise ValueError("Unregistered suite or evaluation/training/dev role overlap")
    if args.awq_recovery_lora_state is None or sha(args.awq_recovery_lora_state) != record["adapter_sha256"]:
        raise ValueError("Candidate adapter SHA mismatch")
    validate_shapes(record["tensor_shapes"], "B3", record["targets"])
    return record
