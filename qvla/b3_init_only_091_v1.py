"""Independent admission for the pre-training B3 SVD factors; never a train gate.

This module deliberately does not change either archived B3 validator.  The
caller must pin the parent artifact SHA in its immutable runtime lock.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "b3-init-only-091-v1-20261016"
MODELS = {"ptq": "A4", "svd-init-only": "B3-S0-091-v1"}
PARAMETERS = 19988480


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1048576), b""):
            digest.update(part)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_metadata(record, spec, training, initial_sha):
    """Pure metadata checks, also usable in the local CPU contract gate."""
    from qvla.extended_peft import target_names, validate_shapes
    if (record.get("version") != "b3-initialization-v1-20261015"
            or record.get("model_id") != "B3"
            or record.get("initialization") != "response-svd"
            or record.get("steps") != 1000
            or record.get("gate") != "PASS_TRAIN_CONTRACT"):
        raise ValueError("Parent must be the completed 046 S artifact, not a renamed S0 artifact")
    if (record.get("training_state_space") != "policy_normalized_proprio"
            or not record.get("zero_output_equal") or not record.get("reload_output_equal")
            or not record.get("smoke_artifact_sha256")
            or not record.get("frozen_before_sha256")
            or record["frozen_before_sha256"] != record.get("frozen_after_sha256")
            or record.get("parameters") != PARAMETERS):
        raise ValueError("Parent frozen/zero/reload/proprio/budget contract is incomplete")
    expected_initial = spec["inherited_cache_sha256"]["initial-adapter.pt"]
    if (initial_sha != expected_initial
            or record.get("reuse_files", {}).get("initial-adapter.pt") != initial_sha
            or record.get("inherited_initial_factor_sha256") != initial_sha):
        raise ValueError("S0 must be the pinned PRE-TRAINING factors, never the final trained adapter")
    if (record.get("teacher_cache_sha256") != spec["inherited_cache_sha256"]["teacher-targets.pt"]
            or spec.get("rank") != 8 or spec.get("alpha_over_rank") != 1
            or spec.get("seed") != 7):
        raise ValueError("Original factor coordinate/scaling/teacher contract changed")
    rows = record["targets"]
    if (len(rows) != 224 or {r["name"] for r in rows} != target_names("B3")
            or record.get("target_sha256") != canonical_sha(rows)):
        raise ValueError("Expected the exact 224 language Linear target descriptors")
    validate_shapes(record["tensor_shapes"], "B3", rows)
    if (sum(8 * sum(r["shape"]) for r in rows) != PARAMETERS
            or any(r.get("parameters") != 8 * sum(r["shape"]) for r in rows)):
        raise ValueError("Descriptor budget differs from fixed rank8 B3")
    samples = training.get("samples", [])
    expected = {(t, r, q) for t in range(10) for r in range(4) for q in (0, 1)}
    actual = {(r["task_id"], r["init_state_index"], r["query_in_episode"]) for r in samples}
    if (len(samples) != 80 or actual != expected
            or training.get("role") != "student_state_train"
            or training.get("source_model_id") != "A4"
            or training.get("state_space") != "policy_normalized_proprio"
            or not training.get("source_run_sha256")):
        raise ValueError("Original A4 training roles/input-state identity changed")


def validate_provenance(args):
    from qvla.b3_initialization_v1 import SOURCES, SPEC
    from qvla.extended_peft import registry
    if sha(args.initial_artifact) != args.initial_artifact_sha256:
        raise ValueError("Parent artifact SHA differs from the explicit trusted plan anchor")
    record = json.loads(args.initial_artifact.read_text(encoding="utf-8"))
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    training = json.loads(args.recovery_training_manifest.read_text(encoding="utf-8"))
    if (record.get("spec_sha256") != sha(SPEC)
            or record.get("training_manifest_sha256") != sha(args.recovery_training_manifest)
            or record.get("source_sha256") != {name: sha(ROOT / name) for name in SOURCES}):
        raise ValueError("Pinned 046 source/spec/training-manifest bytes changed")
    validate_metadata(record, spec, training, sha(args.initial_state))
    hashes = registry()["profile_sha256"]
    if (len(args.profile) != 1 or sha(args.profile[0]) != hashes["w2_g128"]
            or sha(args.awq_primary_group64_profile) != hashes["w2_g64"]):
        raise ValueError("Expected the unchanged A4 G128/G64 quantization profiles")
    if (args.model_id != MODELS[args.candidate]
            or (args.method, args.weight_bits, args.activation_bits, args.awq_candidate)
            != ("awq", 2, 16, "w2-attention-no-clip-primary-g64")
            or (args.task_suite_name, args.seed, args.env_seed, args.seed_protocol)
            != ("libero_spatial", 0, 1, "paired")
            or args.initial_state_offset < 5 or args.num_trials_per_task < 1
            or args.initial_state_offset + args.num_trials_per_task > 50
            or not args.trace_actions or not args.trace_observations):
        raise ValueError("Unregistered candidate, quantization, paired protocol or development range")
    return record


def frozen_digest(modules):
    """Match the archived model/head/proprio stream, excluding only adapter keys."""
    import torch
    digest = hashlib.sha256()
    for prefix, model in modules:
        for name, value in sorted(model.state_dict().items()):
            if prefix == "model" and ".awq_recovery_lora." in name:
                continue
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite frozen tensor: {prefix}.{name}")
            digest.update(f"{prefix}.{name}:{value.dtype}:{tuple(value.shape)}".encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def copy_factors(model, state, zero=False):
    import torch
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            parameter = parameters[name]
            if parameter.shape != value.shape:
                raise ValueError("Runtime adapter shape changed")
            parameter.copy_(torch.zeros_like(parameter) if zero and ".awq_recovery_lora.1." in name
                            else value.to(parameter))


def first_query_contract(model, cfg, observation, task, kwargs, get_action, record,
                         state, state_path):
    """No training; raw observations go through deployed get_action exactly once.

    Probe calls use independent deep copies and restore every RNG stream, so
    they cannot double-normalize the actual rollout observation or reseed it.
    """
    import copy
    import random
    import numpy as np
    import torch
    from qvla.extended_peft import load_state
    from qvla.extended_proprio import prepare_proprio
    raw = copy.deepcopy(observation)
    modules = (("model", model), ("action_head", kwargs["action_head"]),
               ("proprio", kwargs["proprio_projector"]))
    for _name, module in modules:
        module.requires_grad_(False)
        module.eval()
    before = frozen_digest(modules)
    if before != record["frozen_before_sha256"]:
        raise ValueError("Runtime A4/head/proprio bytes differ from the archived frozen base")
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    cpu_rng = torch.get_rng_state()
    try:
        with torch.no_grad():
            reference_observation = copy.deepcopy(raw)
            reference = np.asarray(get_action(cfg, model, reference_observation, task, **kwargs)).copy()
            normalized = prepare_proprio(np.asarray(raw["state"]).copy(),
                model.norm_stats[cfg.unnorm_key]["proprio"], "raw_proprio_before_get_action")
            if not np.array_equal(np.asarray(reference_observation["state"]), normalized):
                raise ValueError("Deployed policy proprio does not match exactly one normalization")
            if state is not None:
                copy_factors(model, state, zero=True)
                zero = np.asarray(get_action(cfg, model, copy.deepcopy(raw), task, **kwargs)).copy()
                reloaded = load_state(state_path, "B3", record["targets"])
                copy_factors(model, reloaded)
            else:
                zero = reference.copy()
            restored = np.asarray(get_action(cfg, model, copy.deepcopy(raw), task, **kwargs)).copy()
        if (reference.shape != (8, 7) or not np.isfinite(reference).all()
                or not np.isfinite(zero).all() or not np.array_equal(reference, restored)):
            raise ValueError("Actual-policy shape/finite/saved-reloaded output mismatch")
        if frozen_digest(modules) != before:
            raise ValueError("First-query gate changed frozen model/head/proprio")
    finally:
        if state is not None:
            copy_factors(model, state)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    return {"gate": "PASS_B3_INIT_ONLY_ACTUAL_MODEL", "version": VERSION,
            "optimization_steps": 0, "frozen_sha256": before,
            "reload_output_equal": True, "all_parameters_frozen": True,
            "raw_state": np.asarray(raw["state"]).tolist(),
            "state_space": "raw_proprio_before_get_action",
            "single_proprio_normalization_equal": True,
            "proprio_path": "deployed get_action on separate raw copies; one normalization per call",
            "reference_action": reference.tolist(), "zero_branch_action": zero.tolist(),
            "zero_branch_comparison": "Pair with canonical A4 on the exact same initial observation",
            "limits": "S0 uses initial factors only; no optimizer steps or closed-loop benefit gate"}
