"""Canonical model names; historical experiment-stage names are not model IDs."""
from __future__ import annotations

import re
import json
import hashlib
from pathlib import Path

W4_LAYERS = {
    "A0": frozenset(range(32)),
    "A1": frozenset(range(8, 24)),
    "A2": frozenset((*range(8, 16), *range(18, 24))),
    "A3": frozenset((*range(8, 16), *range(20, 24))),
    "A4": frozenset(),
}
BASE = {"B0": "A3", "B1": "A3", "B2": "A3", "B3": "A4"}
ADAPTER_LAYERS = {"B0": frozenset((18, 19)),
                  "B1": frozenset((*range(8), *range(16, 20), *range(24, 32)))}
MODEL_IDS = ("BF16", *W4_LAYERS, *BASE)


def validate_training_provenance(args) -> None:
    registry = json.loads((Path(__file__).resolve().parent.parent / "configs/model_registry_v1.json").read_text(encoding="utf-8"))
    artifact = registry["models"][args.model_id]["artifact"]
    if hashlib.sha256(args.awq_recovery_lora_state.read_bytes()).hexdigest() != artifact["adapter_sha256"]:
        raise ValueError("Canonical adapter artifact SHA mismatch; register a new version before changing it")
    if args.recovery_training_manifest is None:
        raise ValueError("Canonical B models require --recovery-training-manifest")
    raw = args.recovery_training_manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != artifact["training_manifest_sha256"]:
        raise ValueError("Canonical training manifest SHA mismatch")
    manifest = json.loads(raw)
    trained = {(r["task_id"], r["init_state_index"]) for r in manifest["samples"]}
    evaluated = {(t, reset) for t in range(10) for reset in range(
        args.initial_state_offset, args.initial_state_offset + args.num_trials_per_task)}
    if args.task_suite_name != "libero_spatial" or trained & evaluated:
        raise ValueError("Canonical Spatial evaluation overlaps training resets or has an unregistered suite")


def validate_profile_provenance(args) -> None:
    if args.model_id is None:
        return
    registry = json.loads((Path(__file__).resolve().parent.parent / "configs/model_registry_v1.json").read_text(encoding="utf-8"))
    hashes = registry["profile_sha256"]
    expected = ([hashes["w4"]] if args.model_id in ("BF16", "A0") else
                [hashes["w2_g128"], hashes["w2_g64"]] + ([] if args.model_id == "A4" else [hashes["w4"]]))
    paths = list(args.profile) + [p for p in (args.awq_primary_group64_profile, args.awq_w4_profile) if p is not None]
    actual = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    if sorted(actual) != sorted(expected):
        raise ValueError("Canonical quantization profile SHA set mismatch")


def validate_request(args) -> None:
    """Reject a name attached to a different recipe before loading the model."""
    name = args.model_id
    if name is None:  # Archived commands remain readable; not canonical evidence.
        return
    if name not in MODEL_IDS:
        raise ValueError(f"Unknown model ID: {name}")
    if name in ("B2", "B3"):
        raise ValueError(f"{name} is preregistered only; implementation/smoke gates remain closed")
    if args.method != "awq" or args.activation_bits != 16 or args.awq_scale_peft_state:
        raise ValueError("Canonical registry requires AWQ A16, without Scale-PEFT")
    has_adapter = args.awq_recovery_lora_state is not None
    if has_adapter != (name in ADAPTER_LAYERS):
        raise ValueError("Model name and Recovery-LoRA presence disagree")
    if name == "BF16":
        if args.awq_scope != "none" or args.awq_candidate != "profile":
            raise ValueError("BF16 must apply zero quantized targets")
        return
    base = BASE.get(name, name)
    layers = frozenset(int(x) for x in args.awq_w4_layers.split(",") if x)
    if args.awq_scope != "all":
        raise ValueError("Canonical A/B models quantify all eligible scopes")
    if base == "A0":
        if args.weight_bits != 4 or args.awq_candidate != "profile" or layers:
            raise ValueError("A0 requires the full W4 profile recipe")
    elif base == "A4":
        if args.weight_bits != 2 or layers or args.awq_candidate != "w2-attention-no-clip-primary-g64":
            raise ValueError("A4 requires language/DINO G64, SigLIP G128, attention no extra clip")
    elif (args.weight_bits != 2 or layers != W4_LAYERS[base] or
          args.awq_candidate != "w2-attention-primary-g64-stage-w4"):
        raise ValueError(f"{base} W4 islands or quantization recipe mismatch")


def validate_adapter_shapes(shapes: dict, model_id: str) -> None:
    """Exact layer/family coverage and rank8; usable without importing torch."""
    layers = ADAPTER_LAYERS[model_id]
    families = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    expected = {f"language_model.model.layers.{i}.{f}.awq_recovery_lora.{j}.weight"
                for i in layers for f in families for j in (0, 1)}
    if set(shapes) != expected:
        raise ValueError(f"{model_id} adapter coverage mismatch")
    for key, shape in shapes.items():
        shape = tuple(shape)
        index = int(re.search(r"awq_recovery_lora\.(\d)\.", key).group(1))
        if len(shape) != 2 or min(shape) <= 0 or shape[0 if index == 0 else 1] != 8:
            raise ValueError(f"{model_id} requires rank8 matrices: {key} {shape}")


def validate_effective_plan(model_id: str | None, targets: dict) -> None:
    if model_id is None:
        return
    base = BASE.get(model_id, model_id)
    if base == "BF16":
        if targets:
            raise ValueError("BF16 unexpectedly has quantized targets")
        return
    language_layers = set()
    vision_counts = {"primary": 0, "fused": 0}
    for name, (_entry, bits, group) in targets.items():
        if name.startswith("language_model."):
            match = re.search(r"\.layers\.(\d+)\.", name)
            if not match:
                raise ValueError(f"Unexpected language target {name}")
            layer = int(match.group(1)); language_layers.add(layer)
            expected_bits = 4 if layer in W4_LAYERS[base] else 2
            if bits != expected_bits or (bits == 2 and group != 64):
                raise ValueError(f"Effective language bits/group mismatch: {name}")
        elif name.startswith("vision_backbone."):
            branch = "fused" if ".fused_featurizer." in name else "primary"
            vision_counts[branch] += 1
            expected_group = 128 if ".fused_featurizer." in name else 64
            if base != "A0" and (bits != 2 or group != expected_group):
                raise ValueError(f"Effective vision bits/group mismatch: {name}")
            if base == "A0" and bits != 4:
                raise ValueError("A0 contains a non-W4 target")
        else:
            raise ValueError(f"Unexpected protected-module quantization: {name}")
    if language_layers != set(range(32)):
        raise ValueError("Canonical recipe must cover all 32 language blocks")
    if len(targets) != 422 or vision_counts != {"primary": 93, "fused": 105}:
        raise ValueError("Canonical connected-422 quantization target coverage mismatch")
