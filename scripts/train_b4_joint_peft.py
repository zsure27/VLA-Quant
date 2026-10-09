"""Isolated B4 joint training entry; smoke10 must pass before train1000.

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
from qvla.b4_joint_peft import (VERSION, sha, canonical_sha, descriptors,
    target_names, student_contract, zero_state, attach_state, load_state, trainable_contract)

from qvla.b4_joint_peft import SOURCES, SPEC, inherited_contract, closure_contract



def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def materials_contract(materials, model_id):
    from scripts.train_extended_peft import materials_contract as archived_contract
    if model_id != "B4": raise ValueError("Joint trainer only accepts B4")
    closure_contract(materials)
    inherited_contract(materials)
    student_contract(materials["student80"], "B4")
    return archived_contract(materials, "B3")


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
    from qvla.run_eval_official_quant import load_profiles
    from qvla.model_registry import validate_effective_plan
    from diagnostics.awq_interventions import attention_no_clip_primary_g64_plan
    from diagnostics.probe import initialize_readonly, predict

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
    plan = attention_no_clip_primary_g64_plan(entries, meta, g64)
    validate_effective_plan("A4", plan[1])
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

    # Reuse exact B3 labels; verify actual BF16 input/label reproducibility in smoke.
    import shutil
    source_dir = args.smoke_dir if prior else Path(materials["b3_train_dir"])
    targets = torch.load(source_dir / "teacher-targets.pt", map_location="cpu", weights_only=True)
    processed = json.loads((source_dir / "teacher-input-hashes.json").read_text())
    if set(targets) != {p.name for p in paths} or set(processed) != set(targets) or any(
            v.shape != (8, 7) or not torch.isfinite(v).all() for v in targets.values()):
        raise ValueError("Inherited teacher coverage/shape/finite mismatch")
    if not prior:
        for path in paths:
            _, normalized, hashes = predict(path, cfg, model, head, proprio, processor,
                                               state_space="policy_normalized_proprio")
            if hashes != processed[path.name] or not torch.equal(normalized, targets[path.name]):
                raise ValueError("Actual B3/B4 teacher inputs or actions differ; stop and audit")
    for name in ("teacher-targets.pt", "teacher-input-hashes.json"):
        shutil.copyfile(source_dir / name, args.output / name)
    for prefix, scales in plan[0].items():
        apply_block_scales(modules[prefix], scales, official)

    # Fixed A4 quantization, same B3 pre-training language factors, new zero visual factors.
    for name, (entry, bits, group) in plan[1].items():
        module = modules[name]
        (apply_llama_entry if name.startswith("language_model.") else apply_awq_entry)(module, entry, official, bits, group)
        if not torch.isfinite(module.weight).all(): raise ValueError(f"Nonfinite quantized weight: {name}")
    base_action = predict(paths[0], cfg, model, head, proprio, processor,
                          state_space="policy_normalized_proprio")[1]
    from qvla.extended_peft import target_names as inherited_targets, load_state as inherited_state
    language = inherited_state(Path(materials["b3_train_dir"]) / "initial-adapter.pt", "B3")
    attach_state(model, language)
    language_action = predict(paths[0], cfg, model, head, proprio, processor,
                              state_space="policy_normalized_proprio")[1]
    if prior:
        initial = load_state(args.smoke_dir / "initial-adapter.pt", "B4", rows)
    else:
        initial = dict(language)
        for name in sorted(inherited_targets("B2")):
            for j, value in enumerate(zero_state(modules[name], name)):
                initial[f"{name}.awq_recovery_lora.{j}.weight"] = value.cpu()
    if any(not torch.equal(initial[k], v) for k, v in language.items()):
        raise ValueError("B4 language initialization differs from B3 pre-training state")
    if any(torch.count_nonzero(initial[f"{n}.awq_recovery_lora.1.weight"]) for n in inherited_targets("B2")):
        raise ValueError("B4 visual initialization is not zero-output")
    # Language is already attached for its reference action. Attach only vision;
    # attaching language twice would duplicate/reject the forward correction.
    vision_names = inherited_targets("B2")
    visual = {k: v for k, v in initial.items() if k.rsplit(".awq_recovery_lora.", 1)[0] in vision_names}
    attach_state(model, visual, vision_names)
    for name in inherited_targets("B3"):
        modules[name].awq_recovery_lora.requires_grad_(True)
    joint_initial_action = predict(paths[0], cfg, model, head, proprio, processor,
                                   state_space="policy_normalized_proprio")[1]
    if not torch.equal(language_action, joint_initial_action):
        raise ValueError("Zero visual branch changed B3 initial action")
    copy_state(model, {k: torch.zeros_like(v) for k, v in initial.items()})
    if not torch.equal(base_action, predict(paths[0], cfg, model, head, proprio, processor,
                                           state_space="policy_normalized_proprio")[1]):
        raise ValueError("Joint zero residual changed A4 action")
    copy_state(model, initial)
    named = trainable_contract(model, args.model_id)
    if sum(p.numel() for p in named.values()) != 26710528:
        raise ValueError("Joint parameter count differs from preregistration")
    torch.save(initial, args.output / "initial-adapter.pt")
    excluded = set(named)
    frozen_modules = (("model", model), ("action_head", head), ("proprio", proprio))
    before = frozen_digest(frozen_modules, excluded)
    from qvla.action_jacobian_batch import load_sample
    from qvla.extended_proprio import prepare_policy_inputs
    cached = [(p.name, *prepare_policy_inputs(load_sample(p), cfg, model, processor,
                                              torch.device("cuda:0"))) for p in paths]
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
    inherited = inherited_contract(materials)
    if args.steps == 1000 and canonical_sha(sequence) != inherited["sample_sequence_sha256"]:
        raise ValueError("B3/B4 sample order or exposure differs")
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
    record = {"version": VERSION, "model_id": args.model_id, "base": "A4",
        "steps": args.steps, "gate": "PASS_SMOKE10" if args.steps == 10 else "PASS_TRAIN_CONTRACT",
        "spec_sha256": sha(SPEC), "source_sha256": source_hashes, "checkpoint_identity": identity,
        "training_manifest_sha256": sha(Path(materials["student80"]) / "manifest.json"),
        "training_state_space": "policy_normalized_proprio",
        "adapter_sha256": sha(state_path), "tensor_shapes": {k: list(v.shape) for k, v in state.items()},
        "targets": rows, "target_sha256": canonical_sha(rows), "zero_output_equal": True, "visual_zero_matches_B3_initial": True,
        "inherited_B3_artifact_sha256": sha(Path(materials["b3_train_dir"]) / "artifact.json"),
        "inherited_reuse_sha256": inherited["reuse_files"],
        "prior_closure_receipt_sha256": sha(materials["prior_closure_receipt"]),
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
    p.add_argument("--model-id", choices=("B4",), required=True)
    p.add_argument("--materials", type=Path, required=True)
    p.add_argument("--steps", type=int, choices=(10, 1000), required=True)
    p.add_argument("--smoke-dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    train(p.parse_args())


if __name__ == "__main__": main()
