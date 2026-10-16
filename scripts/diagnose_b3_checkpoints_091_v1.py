"""Read-only full-training80 diagnostics for the frozen 046 B3 Z/S checkpoints.

One model is loaded: verify BF16 labels against the pinned cache, quantize to
A4 in place, evaluate PTQ and all preregistered checkpoints sequentially. This
is a training-set diagnostic, never a development score or checkpoint selector.
Historical fingerprinted source files and artifacts are not modified.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diagnostics"))

from qvla.extended_peft import canonical_sha, sha

VERSION = "b3-checkpoint-training80-diagnostic-091-v1"
STEPS = (0, 100, 250, 500, 1000)
VARIANTS = ("standard-zero", "response-svd")


def save_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    def encode(value):
        if isinstance(value, (np.ndarray, np.generic)):
            return value.tolist()
        raise TypeError(f"Unsupported JSON value: {type(value)}")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False, default=encode) + "\n", encoding="utf-8")
    temporary.replace(path)


def action_metrics(prediction, teacher, beta=0.1):
    """Aggregate observations equally; axes are observation, chunk step, dimension."""
    p, t = np.asarray(prediction, dtype=np.float64), np.asarray(teacher, dtype=np.float64)
    if p.shape != t.shape or p.ndim != 3 or p.shape[1:] != (8, 7) or not len(p):
        raise ValueError("Require equal nonempty N x 8 x 7 actions")
    if not np.isfinite(p).all() or not np.isfinite(t).all() or beta <= 0:
        raise ValueError("Nonfinite action or invalid SmoothL1 beta")
    delta = p - t
    absolute = np.abs(delta)
    square = delta * delta
    huber = np.where(absolute < beta, 0.5 * square / beta, absolute - 0.5 * beta)
    return {
        "mse": float(square.mean()), "rmse": float(np.sqrt(square.mean())),
        "mae": float(absolute.mean()), "smooth_l1_beta_0_1": float(huber.mean()),
        "max_abs": float(absolute.max()), "bias_per_dim": delta.mean(axis=(0, 1)).tolist(),
        "mse_per_dim": square.mean(axis=(0, 1)).tolist(),
        "rmse_per_dim": np.sqrt(square.mean(axis=(0, 1))).tolist(),
        "mae_per_dim": absolute.mean(axis=(0, 1)).tolist(),
        "mse_per_chunk_step": square.mean(axis=(0, 2)).tolist(),
        "rmse_per_chunk_step": np.sqrt(square.mean(axis=(0, 2))).tolist(),
        "mae_per_chunk_step": absolute.mean(axis=(0, 2)).tolist(),
        "mse_chunk_by_dim": square.mean(axis=0).tolist(),
        "rmse_per_observation": np.sqrt(square.mean(axis=(1, 2))).tolist(),
        "smooth_l1_per_observation": huber.mean(axis=(1, 2)).tolist(),
        "note": "Raw dimensions have different units; pooled raw MSE is descriptive only.",
    }


def low_rank_norms(a, b):
    """Exact Frobenius norm via rank-sized Gram matrices; never materialize B@A."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
        raise ValueError("A rank x input and B output x rank required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite adapter factors")
    product_square = float(np.sum((a @ a.T) * (b.T @ b)))
    if product_square < -1e-8:
        raise ValueError("Unexpected negative product norm")
    return {"A_frobenius": float(np.linalg.norm(a)), "B_frobenius": float(np.linalg.norm(b)),
            "BA_frobenius": float(np.sqrt(max(0.0, product_square))),
            "A_nonzero": int(np.count_nonzero(a)), "B_nonzero": int(np.count_nonzero(b))}


def processed_gripper(raw, process_action, model_family):
    """Invoke the deployed evaluator, including inversion; retain exact zero ties."""
    raw = np.asarray(raw)
    if raw.ndim != 3 or raw.shape[1:] != (8, 7) or not np.isfinite(raw).all():
        raise ValueError("Malformed native raw actions")
    processed = np.stack([
        np.stack([np.asarray(process_action(row.copy(), model_family)) for row in chunk])
        for chunk in raw
    ])
    if processed.shape != raw.shape or not np.isfinite(processed).all():
        raise ValueError("Evaluator action postprocessing returned malformed actions")
    if not np.array_equal(processed[..., :6], raw[..., :6]):
        raise ValueError("Unexpected position/rotation change in gripper-only processor")
    return processed[..., -1]


def gripper_metrics(raw, teacher_raw, process_action, model_family):
    candidate = processed_gripper(raw, process_action, model_family)
    teacher = processed_gripper(teacher_raw, process_action, model_family)
    margin = 2 * np.asarray(raw)[..., -1].astype(np.float64) - 1
    teacher_margin = 2 * np.asarray(teacher_raw)[..., -1].astype(np.float64) - 1
    mismatch = candidate != teacher
    return {
        "decision_disagreement": float(mismatch.mean()), "disagreements": int(mismatch.sum()),
        "decisions": int(mismatch.size), "disagreement_per_chunk_step": mismatch.mean(0).tolist(),
        "candidate_decision": candidate.tolist(), "teacher_decision": teacher.tolist(),
        "candidate_native_raw": np.asarray(raw)[..., -1].tolist(),
        "teacher_native_raw": np.asarray(teacher_raw)[..., -1].tolist(),
        "candidate_prebinarization_signed_margin_2raw_minus1": margin.tolist(),
        "teacher_prebinarization_signed_margin_2raw_minus1": teacher_margin.tolist(),
        "candidate_abs_margin_quantiles": np.quantile(np.abs(margin), [0, .05, .5, .95, 1]).tolist(),
        "teacher_abs_margin_quantiles": np.quantile(np.abs(teacher_margin), [0, .05, .5, .95, 1]).tolist(),
        "candidate_zero_decisions": int(np.count_nonzero(candidate == 0)),
        "teacher_zero_decisions": int(np.count_nonzero(teacher == 0)),
        "margin_scope": "Native model raw action before evaluator sign(2*x-1), not normalized head output; native model may already discretize gripper.",
        "decision_scope": "Actual evaluator process_action including OpenVLA inversion; zero ties remain zero.",
    }


def input_hashes(inputs, state):
    import torch
    result = {name: hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
              for name, value in inputs.items()}
    result["normalized_proprio"] = hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest()
    return result


def checkpoint_contract(materials, old_session, paths):
    """Require the archived full-training pair and exact fixed teacher cache."""
    import torch
    from qvla.b3_initialization_v1 import SOURCES, SPEC, VERSION as TRAIN_VERSION, inherited_cache_contract
    from scripts.audit_b3_initialization_v1 import audit_training

    inherited = inherited_cache_contract(materials)
    pair = audit_training(old_session, 1000)
    source_hashes = {name: sha(ROOT / name) for name in SOURCES}
    expected_names = {p.name for p in paths}
    shared = Path(materials["shared_b3_smoke_dir"])
    teachers = torch.load(shared / "teacher-targets.pt", map_location="cpu", weights_only=True)
    hashes = json.loads((shared / "teacher-input-hashes.json").read_text(encoding="utf-8"))
    if set(teachers) != expected_names or set(hashes) != expected_names or len(expected_names) != 80:
        raise ValueError("Pinned teacher coverage must be exactly the 80 training observations")
    if any(not torch.is_tensor(v) or v.shape != (8, 7) or not torch.isfinite(v).all() for v in teachers.values()):
        raise ValueError("Pinned teacher shape/finite contract failed")
    provenance = {}
    for variant in VARIANTS:
        directory = old_session / variant / "steps-1000"
        record = json.loads((directory / "artifact.json").read_text(encoding="utf-8"))
        if (record.get("version") != TRAIN_VERSION or record.get("model_id") != "B3"
                or record.get("initialization") != variant or record.get("steps") != 1000
                or record.get("gate") != "PASS_TRAIN_CONTRACT"
                or record.get("source_sha256") != source_hashes or record.get("spec_sha256") != sha(SPEC)
                or record.get("training_manifest_sha256") != sha(Path(materials["student80"]) / "manifest.json")
                or record.get("training_state_space") != "policy_normalized_proprio"):
            raise ValueError(f"Frozen training/source/spec contract failed: {variant}")
        for name in ("teacher-targets.pt", "teacher-input-hashes.json", "initial-adapter.pt"):
            if sha(directory / name) != record["reuse_files"][name]:
                raise ValueError(f"Archived training cache SHA changed: {variant}/{name}")
        if (sha(directory / "teacher-targets.pt") != inherited["reuse_files"]["teacher-targets.pt"]
                or sha(directory / "teacher-input-hashes.json") != inherited["reuse_files"]["teacher-input-hashes.json"]
                or sha(directory / "adapter.pt") != record["adapter_sha256"]):
            raise ValueError(f"Archived labels/adapter differ from pinned SHA: {variant}")
        provenance[variant] = {
            "directory": str(directory), "artifact_sha256": sha(directory / "artifact.json"),
            "checkpoint_sha256": {str(step): sha(directory / f"checkpoint-{step:04d}.pt") for step in STEPS},
            "adapter_sha256": record["adapter_sha256"], "record": record,
        }
    return teachers, hashes, {"pair": pair, "inherited_teacher_sha256": inherited["reuse_files"]["teacher-targets.pt"],
                             "training_sources": source_hashes, "variants": provenance}


def tensor_equal_state(left, right):
    import torch
    return set(left) == set(right) and all(torch.equal(left[k], right[k]) for k in left)


def frozen_versions(modules):
    return {f"{prefix}.{name}": value._version
            for prefix, module in modules for name, value in module.state_dict(keep_vars=True).items()
            if ".awq_recovery_lora." not in name}


def run(args):
    import torch
    from diagnostics.awq_interventions import attention_no_clip_primary_g64_plan
    from diagnostics.probe import initialize_readonly
    from experiments.robot.libero import run_libero_eval as evaluator
    from qvla.action_jacobian_batch import load_sample
    from qvla.awq_block import apply_block_scales, apply_llama_entry
    from qvla.baseline_contract import checkpoint_identity
    from qvla.b3_initialization_v1 import SPEC
    from qvla.extended_peft import attach_state, descriptors, load_state, target_names
    from qvla.extended_proprio import prepare_policy_inputs
    from qvla.model_registry import validate_effective_plan
    from qvla.official_quant_adapter import apply_awq_entry, load_official_functions, validate_targets
    from qvla.run_eval_official_quant import load_profiles
    from scripts.train_b3_initialization_v1 import copy_state, materials_contract

    if args.output.exists():
        raise ValueError("Output already exists; diagnostics never overwrite prior results")
    if args.output.resolve().is_relative_to(args.old_session.resolve()):
        raise ValueError("New diagnostic output must be outside the archived training session")
    if not torch.cuda.is_available():
        raise RuntimeError("This entry requires the admitted CUDA runtime")
    started = time.monotonic()
    materials = json.loads(args.materials.read_text(encoding="utf-8"))
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if tuple(spec["checkpoint_steps"]) != STEPS or spec["beta"] != 0.1:
        raise ValueError("Unexpected checkpoint schedule or training loss")
    paths, _calibration, input_audit = materials_contract(materials, "B3")
    teachers, expected_hashes, provenance = checkpoint_contract(materials, args.old_session, paths)
    args.output.mkdir(parents=True)
    manifest = json.loads((Path(materials["student80"]) / "manifest.json").read_text(encoding="utf-8"))
    manifest_rows = {row.get("file", row.get("sample", row.get("name"))): row for row in manifest["samples"]}
    save_json(args.output / "observations.json", {
        "dataset_role": "student_state_train", "state_space": "policy_normalized_proprio",
        "order": [p.name for p in paths], "manifest_sha256": sha(Path(materials["student80"]) / "manifest.json"),
        "samples": [manifest_rows[p.name] for p in paths],
    })
    save_json(args.output / "teacher-input-hashes.json", expected_hashes)
    save_json(args.output / "invocation.json", {
        "version": VERSION, "argv": sys.argv, "materials_sha256": sha(args.materials),
        "old_session": str(args.old_session.resolve()), "script_sha256": sha(__file__),
        "dataset_role": "training80 only; no development/holdout labels", "checkpoint_selection": False,
        "schedule": "BF16 cache verification 80 + A4 PTQ 80 + Z/S five checkpoints x80 = 960 forwards",
        "input_audit": input_audit, "provenance": provenance,
    })
    official = load_official_functions(Path(materials["official_root"]))
    entries, meta = load_profiles([Path(materials["w2_g128"])], "awq")
    plan = attention_no_clip_primary_g64_plan(entries, meta,
                                             load_profiles([Path(materials["w2_g64"])], "awq"))
    validate_effective_plan("B3", plan[1])
    identity = checkpoint_identity(materials["checkpoint"])
    if identity != meta["checkpoint_identity"]:
        raise ValueError("Full checkpoint content differs from AWQ profiles")
    for key, func in (("awq_auto_scale", "awq_auto_scale_block"),
                      ("awq_auto_clip", "awq_auto_clip"), ("awq_quantizer", "awq_quantize")):
        source = inspect.getsourcefile(inspect.unwrap(official[func]))
        if source is None or sha(source) != meta["official_sources"][key]["sha256"]:
            raise ValueError(f"Official quantization source mismatch: {key}")
    cfg, model, head, proprio, processor = initialize_readonly(materials["checkpoint"], spec["seed"])
    if (sha(inspect.getsourcefile(type(model))) != meta["model_source_sha256"]
            or model.language_model.config._attn_implementation != meta["llm_attention"]):
        raise ValueError("Model source/attention differs from calibrated runtime")
    if cfg.model_family != "openvla":
        raise ValueError("Expected the original OpenVLA gripper postprocessing")
    expected_evaluator = Path(materials["oft_root"]) / "experiments/robot/libero/run_libero_eval.py"
    if Path(inspect.getsourcefile(evaluator)).resolve() != expected_evaluator.resolve():
        raise ValueError("Imported evaluator is outside the registered OFT overlay")
    for module in (model, head, proprio):
        module.eval()
        module.requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_grad_enabled(False)
    process = evaluator.process_action
    processing_sources = {}
    for func in (process, process.__globals__["normalize_gripper_action"], process.__globals__["invert_gripper_action"]):
        file = inspect.getsourcefile(func)
        processing_sources[func.__name__] = {"file": file, "file_sha256": sha(file),
                                             "function_source": inspect.getsource(func)}
    save_json(args.output / "action-postprocessing.json", {
        "model_family": cfg.model_family, "sources": processing_sources,
        "native_model_class_source_sha256": meta["model_source_sha256"],
        "action_norm_stats": model.get_action_stats(cfg.unnorm_key),
        "margin_definition": "2 * native_raw_gripper - 1, prior to evaluator binarization/inversion",
    })
    device = torch.device("cuda:0")
    cached = []
    total_cpu_input_bytes = 0
    for path in paths:
        inputs, state = prepare_policy_inputs(load_sample(path), cfg, model, processor, device)
        if input_hashes(inputs, state) != expected_hashes[path.name]:
            raise ValueError(f"Processed inputs differ from teacher cache: {path.name}")
        cpu_inputs = {key: value.detach().cpu() for key, value in inputs.items()}
        total_cpu_input_bytes += sum(value.numel() * value.element_size() for value in cpu_inputs.values())
        cached.append((path.name, cpu_inputs, state.copy()))
        del inputs

    def forward_all(label):
        torch.cuda.synchronize()
        before = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        normalized, raw = [], []
        with torch.no_grad():
            for name, cpu_inputs, state in cached:
                inputs = {key: value.to(device) for key, value in cpu_inputs.items()}
                native_raw, hidden = model.predict_action(
                    **inputs, unnorm_key=cfg.unnorm_key, do_sample=False, proprio=state.copy(),
                    proprio_projector=proprio, action_head=head, noisy_action_projector=None, use_film=False)
                prediction = head.predict_action(hidden).reshape(-1, 7).detach().float().cpu()
                native_raw = (native_raw.detach().cpu().numpy() if torch.is_tensor(native_raw)
                              else np.asarray(native_raw)).copy()
                if (prediction.shape != (8, 7) or native_raw.shape != (8, 7)
                        or not torch.isfinite(prediction).all() or not np.isfinite(native_raw).all()):
                    raise ValueError(f"Malformed/nonfinite predicted action: {label}/{name}")
                if label == "BF16-cache-verification":
                    pinned = teachers[name].detach().cpu().contiguous()
                    if (prediction.dtype != pinned.dtype or not torch.equal(
                            prediction.contiguous().view(torch.uint8), pinned.view(torch.uint8))):
                        raise ValueError(f"Fresh BF16 normalized actions differ bitwise from pinned teacher: {name}")
                normalized.append(prediction.numpy())
                raw.append(native_raw)
                del inputs, native_raw, prediction, hidden
        torch.cuda.synchronize()
        cost = {"wall_seconds": time.monotonic() - before,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                "ending_cuda_allocated_bytes": torch.cuda.memory_allocated(),
                "forward_count": len(cached)}
        print(json.dumps({"case": label, "observations": len(cached), **cost}), flush=True)
        return np.stack(normalized), np.stack(raw), cost

    startup_cost = {"wall_seconds": time.monotonic() - started,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()}
    teacher_normalized, teacher_raw, teacher_cost = forward_all("BF16-cache-verification")
    np.savez_compressed(args.output / "teacher-actions.npz", normalized=teacher_normalized,
                        native_raw=teacher_raw, names=np.asarray([p.name for p in paths]))
    modules = dict(model.named_modules())
    validate_targets(model, list(plan[1]))
    rows = descriptors("B3", modules, plan[1])
    for variant in VARIANTS:
        record = provenance["variants"][variant]["record"]
        if record["checkpoint_identity"] != identity or record["targets"] != rows:
            raise ValueError(f"Actual model/quantization/adapter targets differ: {variant}")
    for prefix, scales in plan[0].items():
        apply_block_scales(modules[prefix], scales, official)
    for name, (entry, bits, group) in plan[1].items():
        module = modules[name]
        (apply_llama_entry if name.startswith("language_model.") else apply_awq_entry)(module, entry, official, bits, group)
        if not torch.isfinite(module.weight).all():
            raise ValueError(f"Quantization produced nonfinite weight: {name}")

    case_records = []
    ptq_actions = None

    def evaluate(label, factor_norms=None, checkpoint_sha=None):
        normalized, raw, cost = forward_all(label)
        if label == "standard-zero-step-0000":
            if ptq_actions is None or not (np.array_equal(normalized, ptq_actions[0])
                                           and np.array_equal(raw, ptq_actions[1])):
                raise ValueError("Zero-output attachment changed A4 PTQ actions on training80")
        record = {
            "case": label, "dataset_role": "training80", "observations": 80,
            "checkpoint_sha256": checkpoint_sha, "normalized_action": action_metrics(normalized, teacher_normalized),
            "native_raw_action": action_metrics(raw, teacher_raw),
            "gripper": gripper_metrics(raw, teacher_raw, process, cfg.model_family),
            "factor_norms": factor_norms, "cost": cost,
        }
        if factor_norms is not None:
            record["factor_summary"] = {
                key: float(np.sqrt(sum(row[key] ** 2 for row in factor_norms.values())))
                for key in ("A_frobenius", "B_frobenius", "BA_frobenius")}
            record["factor_summary"]["max_BA_to_base_ratio"] = max(
                row["BA_to_base_frobenius_ratio"] for row in factor_norms.values()
                if row["BA_to_base_frobenius_ratio"] is not None)
        np.savez_compressed(args.output / f"{label}-actions.npz", normalized=normalized, native_raw=raw,
                            names=np.asarray([p.name for p in paths]))
        save_json(args.output / f"{label}.json", record)
        case_records.append(record)
        save_json(args.output / "progress.json", {"version": VERSION, "completed_cases": [r["case"] for r in case_records],
                                                  "wall_seconds": time.monotonic() - started})
        return normalized, raw

    ptq_actions = evaluate("A4-PTQ-no-adapter")
    initial_path = args.old_session / "standard-zero" / "steps-1000" / "checkpoint-0000.pt"
    state = load_state(initial_path, "B3", rows)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        attach_state(model, state, ())
    del state
    base_modules = (("model", model), ("action_head", head), ("proprio", proprio))
    versions = frozen_versions(base_modules)
    base_norms = {row["name"]: float(modules[row["name"]].weight.detach().float().norm().cpu()) for row in rows}
    for variant in VARIANTS:
        directory = args.old_session / variant / "steps-1000"
        for step in STEPS:
            checkpoint = directory / f"checkpoint-{step:04d}.pt"
            state = load_state(checkpoint, "B3", rows)  # CPU only; no second GPU model/state.
            if step in (0, 1000):
                reference_path = directory / ("initial-adapter.pt" if step == 0 else "adapter.pt")
                reference = load_state(reference_path, "B3", rows)
                if not tensor_equal_state(state, reference):
                    raise ValueError(f"Checkpoint endpoint differs from canonical state: {variant}/{step}")
                del reference
            norms = {}
            for name in sorted(target_names("B3")):
                values = low_rank_norms(state[f"{name}.awq_recovery_lora.0.weight"].float().numpy(),
                                        state[f"{name}.awq_recovery_lora.1.weight"].float().numpy())
                values["quantized_base_weight_frobenius"] = base_norms[name]
                values["BA_to_base_frobenius_ratio"] = values["BA_frobenius"] / base_norms[name] if base_norms[name] else None
                norms[name] = values
            copy_state(model, state)
            del state
            evaluate(f"{variant}-step-{step:04d}", norms, sha(checkpoint))
            if frozen_versions(base_modules) != versions:
                raise ValueError("Frozen model/head/proprio tensor version changed during checkpoint diagnosis")
            gc.collect()
    if len(case_records) != 11 or any(p.grad is not None for module in (model, head, proprio) for p in module.parameters()):
        raise ValueError("Wrong case coverage or unexpected gradient retention")
    summary = {
        "version": VERSION, "gate": "PASS_B3_TRAINING80_CHECKPOINT_DIAGNOSTIC",
        "dataset_role": "Same 80 optimization observations, not held-out development or closed-loop evaluation",
        "teacher_cache_verified_bitwise": True, "processed_inputs_verified": 80,
        "observations_each": 80, "checkpoint_steps": list(STEPS), "forward_count_total": 960,
        "model_loads": 1, "gradient_enabled": False, "base_tensor_versions_unchanged": True,
        "Z_step0_equals_PTQ_all80": True,
        "cached_inputs_cpu_bytes": total_cpu_input_bytes,
        "wall_seconds": time.monotonic() - started, "startup_cost": startup_cost,
        "teacher_verification_cost": teacher_cost,
        "cases": [{"case": row["case"], "checkpoint_sha256": row["checkpoint_sha256"],
                   "normalized_action": row["normalized_action"], "native_raw_action": row["native_raw_action"],
                   "gripper_disagreement": row["gripper"]["decision_disagreement"],
                   "factor_summary": row.get("factor_summary"), "cost": row["cost"]}
                  for row in case_records],
        "limits": ["No endpoint loss is used to select a checkpoint; all five scheduled checkpoints are reported.",
                   "Raw actions are native model outputs, never a differentiable-unnormalization proxy.",
                   "Native model gripper processing may discretize the raw margin; normalized head output is preserved separately.",
                   "A4 PTQ here is an offline training-input reference, not a new closed-loop success baseline.",
                   "BF16 carries fake-quantized weights; this does not measure packed low-bit kernel speed.",
                   "Teacher verification, loading, quantization, factor norms and I/O are separated from forward timing."],
    }
    save_json(args.output / "summary.json", summary)
    save_json(args.output / "SHA256SUMS.json", {p.name: sha(p) for p in sorted(args.output.iterdir()) if p.is_file()})
    print(json.dumps({"gate": summary["gate"], "output": str(args.output),
                      "wall_seconds": summary["wall_seconds"], "forward_count": 960}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materials", type=Path, required=True)
    parser.add_argument("--old-session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
