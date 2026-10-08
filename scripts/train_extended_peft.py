"""Independent B2/B3 training entry; smoke10 must pass before train1000.

Run in the frozen OFT overlay on a user-opened server. Existing probe.py and
the four profile-fingerprinted quantization implementations stay unchanged.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diagnostics"))
from qvla.extended_peft import (VERSION, registry, sha, canonical_sha, descriptors,
    target_names, student_contract, zero_state, attach_state, load_state, trainable_contract)

SOURCES = ("scripts/train_extended_peft.py", "qvla/extended_peft.py", "qvla/recovery_lora.py",
           "qvla/model_registry.py", "qvla/run_eval_official_quant.py", "qvla/data_roles.py",
           "qvla/action_jacobian_batch.py", "diagnostics/probe.py", "diagnostics/low_rank_recovery.py",
           "diagnostics/awq_interventions.py")
SPEC = ROOT / "configs/experiments/b2_b3_v2_proprio_20261008.json"


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def materials_contract(materials, model_id):
    """All roles/bytes checked before model loading; never accept a clone claim."""
    from qvla.data_roles import check_disjoint, audit_training_inputs
    from scripts.audit_peft_calibration80 import audit
    expected = registry()["profile_sha256"]
    for key in ("w2_g128", "w2_g64", "w4"):
        if sha(materials[key]) != expected[key]:
            raise ValueError(f"Profile SHA mismatch: {key}")
    if sha(Path(materials["checkpoint"]) / "config.json") != "bcb688a66f3e94a42311b77d37b7e1484886701f7b2aa8e36d165dfccac92ae6":
        raise ValueError("Checkpoint config SHA mismatch")
    if sha(Path(materials["calibration80"]) / "manifest.json") != "ae493d1fe3e32a5e64fb3658fee19b3528fb1c65ecd14c7f92f14ecb48d13742":
        raise ValueError("Frozen SVD calibration manifest mismatch")
    if sha(materials["trajectory_split"]) not in {
            "d4903a2e9a1ee6b474f5215644ba68d494a8feb89354b3a2a88bbe13a9d48124",
            "3560425653ef4b297a48a3854aae20348898129b6673db1474943e95897606a2"}:
        raise ValueError("Frozen trajectory split mismatch")
    calibration_audit = audit(Path(materials["calibration80"]), Path(materials["trajectory_split"]))
    directory = Path(materials["student80"])
    paths, input_audit = student_contract(directory, model_id)
    calibration = sorted(Path(materials["calibration80"]).glob("sample-*.npz"))
    if len(calibration) != 80:
        raise ValueError("SVD inputs must contain exactly 80 files")
    calibration_audit["content_audit"] = audit_training_inputs(calibration, Path(materials["trajectory_split"]))
    check_disjoint(paths, calibration)
    if model_id == "B2":
        if sha(materials["b1_adapter"]) != registry()["models"]["B1"]["artifact"]["adapter_sha256"]:
            raise ValueError("Frozen B1 state mismatch")
    return paths, calibration, {"training": input_audit, "calibration": calibration_audit}


def frozen_digest(modules, excluded):
    """Stream every frozen parameter AND buffer, including action/proprio/B1."""
    import hashlib
    import torch
    h = hashlib.sha256()
    for prefix, model in modules:
        for name, value in sorted(model.state_dict().items()):
            if prefix == "model" and name in excluded:
                continue
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite frozen tensor: {prefix}.{name}")
            h.update(f"{prefix}.{name}:{value.dtype}:{tuple(value.shape)}".encode())
            h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def copy_state(model, state):
    import torch
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            if name not in params or params[name].shape != value.shape:
                raise ValueError(f"Reload shape/name mismatch: {name}")
            params[name].copy_(value.to(params[name]))


def train(args):
    import numpy as np
    import torch
    from qvla.baseline_contract import checkpoint_identity
    from qvla.official_quant_adapter import load_official_functions, apply_awq_entry, validate_targets
    from qvla.awq_block import apply_block_scales, apply_llama_entry
    from qvla.run_eval_official_quant import load_profiles, load_recovery_lora_state
    from qvla.model_registry import W4_LAYERS, validate_effective_plan
    from diagnostics.awq_interventions import attention_visual_stage_plan, attention_no_clip_primary_g64_plan
    from diagnostics.probe import initialize_readonly, predict, Recorder

    spec = json.loads(SPEC.read_text())
    materials = json.loads(args.materials.read_text())
    paths, calibration, input_audit = materials_contract(materials, args.model_id)
    source_hashes = {p: sha(ROOT / p) for p in SOURCES}
    if args.output.exists():
        raise ValueError("Output exists; never overwrite or repeat a completed stage")
    prior = None
    if args.steps == 1000:
        if args.smoke_dir is None:
            raise ValueError("Full training requires --smoke-dir")
        prior = json.loads((args.smoke_dir / "artifact.json").read_text())
        if (prior.get("steps") != 10 or prior.get("gate") != "PASS_SMOKE10" or
                prior.get("model_id") != args.model_id or prior.get("version") != VERSION or
                prior.get("source_sha256") != source_hashes or prior.get("spec_sha256") != sha(SPEC) or
                prior.get("training_manifest_sha256") != sha(Path(materials["student80"]) / "manifest.json")):
            raise ValueError("Smoke gate changed or belongs to another candidate/data/version")
        for name, expected in prior["reuse_files"].items():
            if sha(args.smoke_dir / name) != expected:
                raise ValueError(f"Smoke reuse SHA mismatch: {name}")
    args.output.mkdir(parents=True)
    save_json(args.output / "input-audit.json", input_audit)
    save_json(args.output / "invocation.json", {"argv": sys.argv, "model_id": args.model_id,
        "steps": args.steps, "materials_sha256": sha(args.materials), "spec_sha256": sha(SPEC),
        "source_sha256": source_hashes})
    torch.set_grad_enabled(False)
    started = time.monotonic()
    official = load_official_functions(Path(materials["official_root"]))
    entries, meta = load_profiles([Path(materials["w2_g128"])], "awq")
    g64 = load_profiles([Path(materials["w2_g64"])], "awq")
    if args.model_id == "B2":
        plan = attention_visual_stage_plan(entries, meta, g64,
            load_profiles([Path(materials["w4"])], "awq"), W4_LAYERS["A3"])
    else:
        plan = attention_no_clip_primary_g64_plan(entries, meta, g64)
    validate_effective_plan(args.model_id, plan[1])
    identity = checkpoint_identity(materials["checkpoint"])
    if identity != meta["checkpoint_identity"]:
        raise ValueError("Full checkpoint content differs from profiles")
    for key, func in (("awq_auto_scale", "awq_auto_scale_block"), ("awq_auto_clip", "awq_auto_clip"),
                      ("awq_quantizer", "awq_quantize")):
        # AWQ wraps some functions with torch.no_grad; hash the unwrapped official source.
        source = inspect.getsourcefile(inspect.unwrap(official[func]))
        if source is None or sha(source) != meta["official_sources"][key]["sha256"]:
            raise ValueError(f"Official source SHA mismatch: {key}")
    cfg, model, head, proprio, processor = initialize_readonly(materials["checkpoint"], spec["seed"])
    if sha(inspect.getsourcefile(type(model))) != meta["model_source_sha256"]:
        raise ValueError("Loaded model class differs from profile")
    if model.language_model.config._attn_implementation != meta["llm_attention"]:
        raise ValueError("Attention implementation changed")
    torch.backends.cuda.matmul.allow_tf32 = False
    modules = dict(model.named_modules())
    validate_targets(model, list(plan[1]))
    rows = descriptors(args.model_id, modules, plan[1])
    save_json(args.output / "targets.json", rows)
    if prior and (prior["checkpoint_identity"] != identity or prior["targets"] != rows):
        raise ValueError("Actual backbone/target descriptors differ from smoke")

    # BF16 teacher queried before any quantization or adapter on the SAME observations.
    if prior:
        targets = torch.load(args.smoke_dir / "teacher-targets.pt", map_location="cpu", weights_only=True)
        save_json(args.output / "teacher-input-hashes.json", json.loads((args.smoke_dir / "teacher-input-hashes.json").read_text()))
    else:
        targets = {}
        processed = {}
        for path in paths:
            raw, normalized, hashes = predict(path, cfg, model, head, proprio, processor,
                                               state_space="policy_normalized_proprio")
            if raw.shape != (8, 7) or normalized.shape != (8, 7) or not torch.isfinite(raw).all() or not torch.isfinite(normalized).all():
                raise ValueError("BF16 same-observation teacher must be finite 8x7")
            targets[path.name] = normalized
            processed[path.name] = hashes
        save_json(args.output / "teacher-input-hashes.json", processed)
    if set(targets) != {p.name for p in paths} or any(v.shape != (8, 7) or not torch.isfinite(v).all() for v in targets.values()):
        raise ValueError("Teacher target coverage/shape/finite mismatch")
    torch.save(targets, args.output / "teacher-targets.pt")
    for prefix, scales in plan[0].items():
        apply_block_scales(modules[prefix], scales, official)

    # B3 Response-SVD sees only peft_train action-token activations in scaled BF16 coordinates.
    spool = args.output / "svd-inputs"
    if args.model_id == "B3" and not prior:
        import shutil
        # Reserve measured BF16 activation bytes plus 6 GiB for file overhead and
        # later stage outputs. The fixed 24 GiB check rejected otherwise safe clones.
        anticipated = len(calibration) * 56 * 2 * sum(
            modules[name].in_features for name in target_names("B3"))
        available = shutil.disk_usage(args.output).free
        required = anticipated + 6 * 1024**3
        save_json(args.output / "svd-storage-budget.json", {
            "anticipated_spool_bytes": anticipated, "reserve_bytes": 6 * 1024**3,
            "required_free_bytes": required, "available_free_bytes": available})
        if available < required:
            raise ValueError("Insufficient persistent space for measured BF16 SVD spool plus reserve")
        spool.mkdir()
        recorder = Recorder(model, capture_traces=False)
        handles, calls = [], {}
        index = 0
        def capture(name):
            def hook(_module, inputs):
                x = inputs[0].detach()
                start, count = recorder.action_start, recorder.action_count
                if (start is None or count != 56 or x.ndim != 3 or x.shape[0] != 1
                        or start + count > x.shape[1] or x.dtype != torch.bfloat16):
                    raise ValueError("Real action-token slice unavailable")
                x = x[:, start:start + count].reshape(-1, x.shape[-1]).cpu()
                if not torch.isfinite(x).all():
                    raise ValueError("Nonfinite SVD input")
                key = sha_name(name)
                directory = spool / key
                directory.mkdir(exist_ok=True)
                destination = directory / f"{index:03d}.pt"
                if destination.exists():
                    raise ValueError("SVD target executed more than once per calibration query")
                torch.save(x, destination)
                calls[name] = calls.get(name, 0) + 1
            return hook
        for name in sorted(target_names("B3")):
            handles.append(modules[name].register_forward_pre_hook(capture(name)))
        try:
            for index, path in enumerate(calibration):
                recorder.reset(None)
                predict(path, cfg, model, head, proprio, processor)
        finally:
            for handle in handles:
                handle.remove()
            recorder.close()
        if set(calls) != target_names("B3") or any(c != 80 for c in calls.values()):
            raise ValueError("SVD calibration call coverage mismatch")

    initial = load_state(args.smoke_dir / "initial-adapter.pt", args.model_id, rows) if prior else {}
    for name, (entry, bits, group) in plan[1].items():
        module = modules[name]
        original = module.weight.detach().clone() if args.model_id == "B3" and name in target_names("B3") and not prior else None
        (apply_llama_entry if name.startswith("language_model.") else apply_awq_entry)(module, entry, official, bits, group)
        if not torch.isfinite(module.weight).all():
            raise ValueError(f"Quantization produced nonfinite weights: {name}")
        if original is not None:
            from diagnostics.low_rank_recovery import attach_residual
            inputs = torch.cat([torch.load(p, map_location="cpu", weights_only=True).float()
                                for p in sorted((spool / sha_name(name)).glob("*.pt"))])
            attach_residual(module, original, 8, int(sha_name(name)[:8], 16), input_rows=inputs, init="response-svd")
            for j in (0, 1):
                initial[f"{name}.awq_recovery_lora.{j}.weight"] = module.awq_recovery_lora[j].weight.detach().cpu().clone()
            del inputs, original
    if args.model_id == "B2":
        b1 = load_recovery_lora_state(Path(materials["b1_adapter"]), set(registry()["models"]["B1"]["language_adapter_blocks"]))
        attach_state(model, b1)
    if args.model_id == "B2" and not prior:
        for name in sorted(target_names("B2")):
            for j, value in enumerate(zero_state(modules[name], name)):
                initial[f"{name}.awq_recovery_lora.{j}.weight"] = value.cpu()

    # Actual-model zero residual equality, including B3 before restoring its nonzero SVD init.
    if args.model_id == "B3" and not prior:
        copy_state(model, {k: torch.zeros_like(v) for k, v in initial.items()})
        zero_with_branch = predict(paths[0], cfg, model, head, proprio, processor,
                                   state_space="policy_normalized_proprio")[1]
        hooks = {n: modules[n]._forward_hooks.copy() for n in target_names("B3")}
        try:
            for n in hooks: modules[n]._forward_hooks.clear()
            zero_reference = predict(paths[0], cfg, model, head, proprio, processor,
                                     state_space="policy_normalized_proprio")[1]
        finally:
            for n, saved_hooks in hooks.items(): modules[n]._forward_hooks.update(saved_hooks)
        copy_state(model, initial)
    else:
        zero_reference = predict(paths[0], cfg, model, head, proprio, processor,
                                 state_space="policy_normalized_proprio")[1]
        attach_state(model, initial, target_names(args.model_id))
        saved = {k: v.clone() for k, v in initial.items()}
        copy_state(model, {k: torch.zeros_like(v) for k, v in initial.items()})
        zero_with_branch = predict(paths[0], cfg, model, head, proprio, processor,
                                   state_space="policy_normalized_proprio")[1]
        copy_state(model, saved)
    if not torch.equal(zero_reference, zero_with_branch):
        raise ValueError("Actual-model zero residual output is not exactly equal")
    # Existing SVD attachments are trainable; frozen B1 never becomes trainable.
    for n in target_names(args.model_id):
        modules[n].awq_recovery_lora.requires_grad_(True)
    named = trainable_contract(model, args.model_id)
    torch.save(initial, args.output / "initial-adapter.pt")
    excluded = set(named)
    frozen_modules = (("model", model), ("action_head", head), ("proprio", proprio))
    before = frozen_digest(frozen_modules, excluded)
    from qvla.action_jacobian_batch import load_sample, prepare_inputs
    cached = [(p.name, *prepare_inputs(load_sample(p), cfg, model, processor, torch.device("cuda:0"),
                                       state_space="policy_normalized_proprio")) for p in paths]
    optimizer = torch.optim.AdamW(list(named.values()), lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
    generator = torch.Generator(device="cpu").manual_seed(spec["order_seed"])
    sequence, trace = [], []
    order = []
    with torch.enable_grad():
        for step in range(args.steps):
            if step % len(paths) == 0:
                order = torch.randperm(len(paths), generator=generator).tolist()
            name, inputs, state = cached[order[step % len(paths)]]
            optimizer.zero_grad(set_to_none=True)
            _, hidden = model.predict_action(**inputs, unnorm_key=cfg.unnorm_key, do_sample=False,
                proprio=state, proprio_projector=proprio, action_head=head, noisy_action_projector=None, use_film=False)
            prediction = head.predict_action(hidden).reshape(-1, 7).float()
            if prediction.shape != (8, 7) or not torch.isfinite(prediction).all():
                raise ValueError("Nonfinite/malformed training action")
            loss = torch.nn.functional.smooth_l1_loss(prediction, targets[name].cuda().float(), beta=spec["beta"])
            if not torch.isfinite(loss): raise ValueError("Nonfinite loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in named.values()):
                raise ValueError("Missing or nonfinite gradient on an expected adapter parameter")
            norm = torch.nn.utils.clip_grad_norm_(list(named.values()), spec["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in named.values()): raise ValueError("Nonfinite updated adapter")
            sequence.append(name)
            if step == 0 or (step + 1) % 10 == 0:
                trace.append({"step": step + 1, "loss": float(loss), "gradient_norm": float(norm), "sample": name})
            if (step + 1) % 100 == 0 or step + 1 == args.steps:
                temporary = args.output / "progress-adapter.tmp"
                torch.save({n: p.detach().cpu() for n, p in named.items()}, temporary)
                temporary.replace(args.output / "progress-adapter.pt")
                save_json(args.output / "progress.json", {"completed_steps": step + 1,
                    "source_sha256": source_hashes, "spec_sha256": sha(SPEC),
                    "note": "Safety checkpoint, not a canonical artifact or automatic resume approval"})
                print(json.dumps(trace[-1]), flush=True)
            if time.monotonic() - started > spec["max_stage_seconds"]:
                raise TimeoutError("Preregistered stage time budget exhausted")
    after = frozen_digest(frozen_modules, excluded)
    if before != after: raise ValueError("Frozen base/head/proprio/B1 changed during training")
    state = {n: p.detach().cpu().clone() for n, p in named.items()}
    state_path = args.output / "adapter.pt"
    torch.save(state, state_path)
    reference = predict(paths[0], cfg, model, head, proprio, processor,
                        state_space="policy_normalized_proprio")[1]
    reloaded = load_state(state_path, args.model_id, rows)
    copy_state(model, {k: torch.zeros_like(v) for k, v in state.items()})
    copy_state(model, reloaded)
    if not torch.equal(reference, predict(paths[0], cfg, model, head, proprio, processor,
                                          state_space="policy_normalized_proprio")[1]):
        raise ValueError("Actual-model saved/reloaded adapter action mismatch")
    record = {"version": VERSION, "model_id": args.model_id, "base": "A3" if args.model_id == "B2" else "A4",
        "steps": args.steps, "gate": "PASS_SMOKE10" if args.steps == 10 else "PASS_TRAIN_CONTRACT",
        "spec_sha256": sha(SPEC), "source_sha256": source_hashes, "checkpoint_identity": identity,
        "training_manifest_sha256": sha(Path(materials["student80"]) / "manifest.json"),
        "training_state_space": "policy_normalized_proprio",
        "adapter_sha256": sha(state_path), "tensor_shapes": {k: list(v.shape) for k, v in state.items()},
        "targets": rows, "target_sha256": canonical_sha(rows), "zero_output_equal": True,
        "reload_output_equal": True, "frozen_before_sha256": before, "frozen_after_sha256": after,
        "parameters": sum(p.numel() for p in named.values()), "serialized_bytes": state_path.stat().st_size,
        "sample_sequence_sha256": canonical_sha(sequence), "trace": trace,
        "training_seconds": time.monotonic() - started, "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "reuse_files": {n: sha(args.output / n) for n in ("initial-adapter.pt", "teacher-targets.pt", "teacher-input-hashes.json")},
        "smoke_artifact_sha256": sha(args.smoke_dir / "artifact.json") if prior else None,
        "interpretation": "Training/structural evidence only; no closed-loop benefit claim"}
    save_json(args.output / "artifact.json", record)
    print(json.dumps({k: record[k] for k in ("model_id", "steps", "gate", "parameters", "training_seconds")}))


def sha_name(name):
    import hashlib
    return hashlib.sha256(name.encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", choices=("B2", "B3"), required=True)
    p.add_argument("--materials", type=Path, required=True)
    p.add_argument("--steps", type=int, choices=(10, 1000), required=True)
    p.add_argument("--smoke-dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    train(p.parse_args())


if __name__ == "__main__": main()
