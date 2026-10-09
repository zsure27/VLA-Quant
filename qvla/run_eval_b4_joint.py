# B4 v1 evaluator fork: preserve archived evaluator/profile source hashes.
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qvla.baseline_contract import CONTRACT_ID, check_scope, checkpoint_identity
from qvla.reproducibility import seed_all

from qvla.official_quant_adapter import (
    PROFILE_FORMAT,
    apply_awq_entry,
    apply_smoothquant_smoothing,
    load_official_functions,
    make_smoothquant_activation_hook,
    scope_counts,
    source_sha256,
    smoothquant_weight,
    validate_targets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate official-source AWQ or SmoothQuant on OpenVLA."
    )
    parser.add_argument("--method", required=True, choices=("awq", "smoothquant"))
    parser.add_argument("--weight-bits", required=True, type=int, choices=(2, 4, 8, 16))
    parser.add_argument("--activation-bits", required=True, type=int, choices=(4, 8, 16))
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--profile", required=True, type=Path, action="append")
    parser.add_argument("--official-root", required=True, type=Path)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--num_trials_per_task", type=int, default=2)
    parser.add_argument("--initial-state-offset", type=int, default=0)
    parser.add_argument("--a1-reset-dir", type=Path,
                        help="Frozen per-task .npy reset arrays for preregistered A1; bypasses demo-success JSON")
    parser.add_argument("--local_log_dir", required=True)
    parser.add_argument("--libero_root", default=os.environ.get("LIBERO_ROOT", ""))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument("--seed-protocol", choices=("upstream", "paired"), default="upstream")
    parser.add_argument("--awq-scope", choices=("all", "language", "vision", "none"), default="all",
                        help="Diagnostic ablation only; none is the same-loader BF16 control")
    parser.add_argument("--trace-actions", action="store_true", help="Save raw policy chunks and query-time proprio for diagnostics")
    parser.add_argument(
        "--trace-observations", action="store_true",
        help="P2.5: save observations actually visited by the student, its raw chunk, and causal H17 summaries",
    )
    parser.add_argument(
        "--awq-candidate",
        choices=("profile", "w2-no-clip-primary-g64", "w2-attention-no-clip-primary-g64", "w2-attention-no-clip-dino-g128-siglip-g64", "w2-attention-no-clip-visual-g128", "w2-attention-primary-g64-stage-w4", "w2-no-clip-stage-w4", "w2-fixed-coordinates-stage-w4"),
        default="profile",
        help="AWQ rollout recipe; the tuned W2 candidate uses language no-clip, primary vision G64, fused vision G128",
    )
    parser.add_argument(
        "--awq-primary-group64-profile",
        type=Path,
        help="independent W2/G64 profile required by a primary-g64 candidate",
    )
    parser.add_argument("--awq-w4-profile", type=Path)
    parser.add_argument("--awq-w4-layers", default="")
    parser.add_argument("--awq-scale-peft-state", type=Path)
    parser.add_argument("--awq-recovery-lora-state", type=Path)
    parser.add_argument("--awq-visual-lora-state", type=Path,
                        help="B2 only: visual increment, separately validated from frozen B1")
    parser.add_argument("--extended-peft-manifest", type=Path,
                        help="Versioned B2/B3 artifact with smoke, train, target and data hashes")
    from qvla.b4_joint_peft import MODEL_IDS, validate_request
    parser.add_argument("--model-id", choices=MODEL_IDS,
                        help="Canonical registry identity; omit only for archived/diagnostic recipes")
    parser.add_argument("--recovery-training-manifest", type=Path,
                        help="Canonical B0/B1 training provenance; evaluation must exclude its resets")
    args = parser.parse_args()
    validate_request(args)
    from qvla.b4_joint_peft import validate_profile_provenance
    validate_profile_provenance(args)
    if args.model_id in ("B0", "B1"):
        from qvla.model_registry import validate_training_provenance
        validate_training_provenance(args)
    if args.model_id in ("B2", "B4"):
        from qvla.b4_joint_peft import validate_artifact
        validate_artifact(args)
    elif args.awq_visual_lora_state is not None or args.extended_peft_manifest is not None:
        parser.error("Extended adapter flags require canonical B2/B3")
    if args.awq_scope != "all" and args.method != "awq":
        parser.error("Scoped diagnostics are currently supported only for AWQ")
    primary_g64_candidates = {
        "w2-no-clip-primary-g64",
        "w2-attention-no-clip-primary-g64",
        "w2-attention-no-clip-dino-g128-siglip-g64",
        "w2-attention-no-clip-visual-g128",
        "w2-attention-primary-g64-stage-w4",
    }
    if (args.awq_primary_group64_profile is not None) != (args.awq_candidate in primary_g64_candidates):
        parser.error("primary-g64 candidates require exactly one --awq-primary-group64-profile")
    if args.initial_state_offset < 0 or args.num_trials_per_task < 1:
        parser.error("Invalid initial-state shard range")
    if args.initial_state_offset and args.seed_protocol != "paired":
        parser.error("Shard offset requires paired per-episode seeding")
    if args.a1_reset_dir is not None:
        if args.seed_protocol != "paired" or args.task_suite_name != "libero_spatial":
            parser.error("A1 reset arrays require paired LIBERO Spatial evaluation")
        reset_dir = args.a1_reset_dir.resolve()
        for task_id in range(10):
            reset_file = reset_dir / f"task-{task_id:02d}.npy"
            if not reset_file.is_file():
                parser.error(f"Missing A1 reset array: {reset_file}")
            states = np.load(reset_file, allow_pickle=False, mmap_mode="r")
            if (states.ndim != 2 or
                    states.shape[0] < args.initial_state_offset + args.num_trials_per_task or
                    not np.isfinite(states).all()):
                parser.error(f"Invalid A1 reset array or shard range: {reset_file}")
        args.a1_reset_dir = reset_dir
    if args.awq_candidate == "w2-attention-primary-g64-stage-w4":
        if (args.method, args.weight_bits, args.activation_bits, args.awq_scope) != ("awq", 2, 16, "all"):
            parser.error("Combined stage rescue requires full-scope AWQ W2A16")
        if not args.awq_w4_layers or args.awq_w4_profile is None or args.awq_primary_group64_profile is None:
            parser.error("Combined stage rescue requires G64 and W4 peer profiles plus layers")
    elif args.awq_candidate == "w2-no-clip-stage-w4":
        if (args.method, args.weight_bits, args.activation_bits, args.awq_scope) != ("awq", 2, 16, "language"):
            parser.error("Stage W4 rescue requires language-only AWQ W2A16")
        if not args.awq_w4_layers or args.awq_w4_profile is None or args.awq_primary_group64_profile is not None:
            parser.error("Stage W4 rescue requires layers and a W4 peer, without a vision G64 peer")
    elif args.awq_candidate == "w2-fixed-coordinates-stage-w4":
        if (args.method, args.weight_bits, args.activation_bits, args.awq_scope) != ("awq", 2, 16, "language"):
            parser.error("Fixed-coordinate rescue requires language-only AWQ W2A16")
        if not args.awq_w4_layers or args.awq_w4_profile is not None or args.awq_primary_group64_profile is not None:
            parser.error("Fixed-coordinate rescue requires layers without any peer profile")
    elif args.awq_w4_layers or args.awq_w4_profile is not None:
        parser.error("W4 stage parameters require --awq-candidate w2-no-clip-stage-w4")
    if args.awq_scale_peft_state is not None and args.awq_recovery_lora_state is not None:
        parser.error("Select one PEFT state per paired evaluation")
    if args.awq_scale_peft_state is not None or args.awq_recovery_lora_state is not None:
        is_b3 = args.model_id == "B4"
        if not is_b3 and args.awq_candidate != "w2-attention-primary-g64-stage-w4":
            parser.error("PEFT state requires the exact attention/visual stage backbone")
        layers = {int(value) for value in args.awq_w4_layers.split(",") if value}
        if not is_b3 and layers != set(range(8, 16)) | set(range(20, 24)):
            parser.error("PEFT state is pinned to the exact 12L backbone")
        state_path = args.awq_scale_peft_state or args.awq_recovery_lora_state
        if not state_path.is_file():
            parser.error("PEFT state file does not exist")
    return args


def load_scale_peft_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    families = (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    )
    expected = {
        f"language_model.model.layers.{layer}.{family}.awq_scale_peft.log_step_residual"
        for layer in (18, 19) for family in families
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        missing = sorted(expected - set(payload)) if isinstance(payload, dict) else sorted(expected)
        extra = sorted(set(payload) - expected) if isinstance(payload, dict) else []
        raise RuntimeError(f"Scale-PEFT state key mismatch: missing={missing}, extra={extra}")
    for name, value in payload.items():
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise RuntimeError(f"Invalid Scale-PEFT tensor: {name}")
    return payload


RECOVERY_LORA_ALLOWED_LANGUAGE_W2_LAYERS = frozenset(
    (*range(0, 8), *range(16, 20), *range(24, 32))
)


def load_recovery_lora_state(
    path: Path,
    expected_layers: set[int] | frozenset[int] | None = None,
) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    key_pattern = re.compile(
        r"^language_model\.model\.layers\.(\d+)\."
        r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
        r"mlp\.(?:gate_proj|up_proj|down_proj))\."
        r"awq_recovery_lora\.[01]\.weight$"
    )
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError("Recovery-LoRA state must be a non-empty tensor mapping")
    target_layers: set[int] = set()
    for name in payload:
        match = key_pattern.fullmatch(name) if isinstance(name, str) else None
        if match is None:
            raise RuntimeError(f"Invalid Recovery-LoRA state key: {name}")
        target_layers.add(int(match.group(1)))
    if not target_layers <= RECOVERY_LORA_ALLOWED_LANGUAGE_W2_LAYERS:
        invalid = sorted(target_layers - RECOVERY_LORA_ALLOWED_LANGUAGE_W2_LAYERS)
        raise RuntimeError(f"Recovery-LoRA may target only language W2 blocks: {invalid}")
    if expected_layers is not None and target_layers != set(expected_layers):
        raise RuntimeError(
            f"Recovery-LoRA layer mismatch: expected={sorted(expected_layers)}, "
            f"actual={sorted(target_layers)}"
        )
    families = (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    )
    expected = {
        f"language_model.model.layers.{layer}.{family}.awq_recovery_lora.{index}.weight"
        for layer in target_layers for family in families for index in (0, 1)
    }
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise RuntimeError(f"Recovery LoRA state key mismatch: missing={missing}, extra={extra}")
    for name, value in payload.items():
        if not isinstance(value, torch.Tensor) or not value.is_floating_point() or not torch.isfinite(value).all():
            raise RuntimeError(f"Invalid Recovery LoRA tensor: {name}")
    return payload


def load_profiles(paths: list[Path], method: str) -> tuple[dict[str, Any], dict[str, Any]]:
    entries: dict[str, Any] = {}
    metadata: dict[str, Any] | None = None
    common_fields = (
        "format_version", "method", "bits", "activation_bits", "checkpoint",
        "num_samples", "sample_names", "group_size", "smooth_alpha", "official_sources",
        "contract_id", "checkpoint_identity", "sample_sha256", "seed", "algorithm", "llm_attention",
        "implementation_sources", "model_source_sha256", "calibration_manifest",
    )
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != PROFILE_FORMAT:
            raise RuntimeError(f"Unsupported profile format in {path}")
        if payload.get("method") != method:
            raise RuntimeError(f"Profile method mismatch in {path}")
        if payload.get("contract_id") != CONTRACT_ID or not payload.get("checkpoint_identity") or not payload.get("sample_sha256"):
            raise RuntimeError("缺少检查点/样本指纹，请重新校准，不能仅修改旧 metadata 冒充新 profile")
        if method == "awq" and payload.get("algorithm") != "awq_official_llama_vision_adapter_v1":
            raise RuntimeError("逐 Linear 实验不是官方 Llama block AWQ，禁止用于正式基线")
        if method == "awq" and len(paths) != 1:
            raise RuntimeError("block AWQ 使用一个完整 profile，禁止拼接独立搜索的 shard")
        overlap = set(entries).intersection(payload["entries"])
        if overlap:
            raise RuntimeError(f"Duplicate profile targets: {sorted(overlap)[:5]}")
        entries.update(payload["entries"])
        if metadata is None:
            metadata = payload
        else:
            for field in common_fields:
                if payload.get(field) != metadata.get(field):
                    raise RuntimeError(f"Profile metadata mismatch for {field} in {path}")
    if metadata is None or not entries:
        raise RuntimeError("No quantization profiles loaded")
    for name in ("official_quant_adapter.py", "awq_block.py", "action_jacobian_batch.py", "runtime_contract.py"):
        expected = metadata.get("implementation_sources", {}).get(name)
        if expected != source_sha256(Path(__file__).with_name(name)):
            raise RuntimeError(f"适配代码已变化，请重新校准：{name}")
    check_scope(entries)
    return entries, metadata


def select_awq_scope(names: list[str], scope: str) -> list[str]:
    check_scope(names)
    if scope == "all":
        return names
    if scope == "none":
        return []
    if scope not in ("language", "vision"):
        raise ValueError(f"Unknown AWQ diagnostic scope: {scope}")
    return [name for name in names if name.startswith("language_model.") == (scope == "language")]


@torch.no_grad()
def apply_quantization(
    model: torch.nn.Module,
    entries: dict[str, Any],
    metadata: dict[str, Any],
    method: str,
    official: dict[str, Any],
    weight_bits: int,
    activation_bits: int,
    awq_plan: tuple[dict[str, Any], dict[str, tuple[dict[str, Any], int, int]], list[str]] | None = None,
    awq_scope: str = "all",
    scale_peft_state: dict[str, torch.Tensor] | None = None,
    recovery_lora_state: dict[str, torch.Tensor] | None = None,
) -> list[Any]:
    names = list(awq_plan[1] if awq_plan is not None else entries)
    check_scope(names)
    if method != "awq" and awq_scope != "all":
        raise ValueError("Scoped diagnostics are supported only for AWQ")
    if method == "awq":
        names = select_awq_scope(names, awq_scope)
        if not names:
            print("[diagnostic] same-loader BF16 control: no scales, clipping or weight quantization applied")
            return []
    modules = validate_targets(model, names)
    for name, module in modules.items():
        entry = awq_plan[1][name][0] if awq_plan is not None else entries[name]
        if tuple(entry.get("shape", ())) != tuple(module.weight.shape):
            raise RuntimeError(f"Profile weight shape mismatch for {name}")
        tensors = [entry.get("input_scale"), entry.get("clip_max"), entry.get("activation_absmax")]
        for value in tensors:
            if value is not None and not torch.isfinite(value).all():
                raise RuntimeError(f"Non-finite profile tensor for {name}")
    handles = []
    if method == "awq":
        if weight_bits not in (2, 4, 8) or activation_bits != 16 or metadata["bits"] != weight_bits:
            raise ValueError("AWQ 仅 W2/W4/W8 A16；必须按当前位宽重新搜索 scale/clip")
        from qvla.awq_block import apply_block_scales, apply_llama_entry
        all_modules = dict(model.named_modules())
        expected_blocks = {f"language_model.model.layers.{i}" for i in range(32)}
        block_scales = awq_plan[0] if awq_plan is not None else metadata.get("block_scales", {})
        if set(block_scales) != expected_blocks:
            raise ValueError("官方 block scale 不完整，禁止隐式退化到 RTN")
        if awq_scope != "vision":
            for prefix, scales in block_scales.items():
                apply_block_scales(all_modules[prefix], scales, official)
        for index, name in enumerate(names):
            entry, target_bits, target_group = (
                awq_plan[1][name]
                if awq_plan is not None
                else (entries[name], weight_bits, int(metadata.get("group_size", 128)))
            )
            apply_entry = apply_llama_entry if name.startswith("language_model.") else apply_awq_entry
            state_key = f"{name}.awq_scale_peft.log_step_residual"
            teacher_weight = (
                modules[name].weight.detach().clone() if scale_peft_state is not None and state_key in scale_peft_state else None
            )
            if name.startswith("language_model.") and entry.get("recipe") != "official_llama_block_v1":
                raise ValueError("LLM profile 不是官方 block 搜索")
            apply_entry(
                modules[name],
                entry,
                official,
                bits=target_bits,
                group_size=target_group,
            )
            if teacher_weight is not None:
                from qvla.fixed_code_scale import attach_fixed_code_scale_state
                details = attach_fixed_code_scale_state(
                    modules[name], teacher_weight, entry, target_bits, target_group,
                    scale_peft_state[state_key],
                )
                print("[scale-peft] " + json.dumps({"target": name, **details}, sort_keys=True))
            lora_prefix = f"{name}.awq_recovery_lora"
            if recovery_lora_state is not None and f"{lora_prefix}.0.weight" in recovery_lora_state:
                from qvla.recovery_lora import attach_recovery_lora_state
                details = attach_recovery_lora_state(
                    modules[name],
                    recovery_lora_state[f"{lora_prefix}.0.weight"],
                    recovery_lora_state[f"{lora_prefix}.1.weight"],
                )
                print("[recovery-lora] " + json.dumps({"target": name, **details}, sort_keys=True))
            print(f"[official-awq] {index + 1}/{len(names)} {name}")
        if scale_peft_state is not None:
            attached = {
                f"{name}.awq_scale_peft.log_step_residual" for name in names
                if hasattr(modules[name], "awq_scale_peft")
            }
            if attached != set(scale_peft_state):
                raise RuntimeError("Not all Scale-PEFT state tensors were attached")
        if recovery_lora_state is not None:
            attached = {
                f"{name}.awq_recovery_lora.{index}.weight"
                for name in names if hasattr(modules[name], "awq_recovery_lora") for index in (0, 1)
            }
            if attached != set(recovery_lora_state):
                raise RuntimeError("Not all Recovery LoRA state tensors were attached")
    else:
        if weight_bits not in (4, 8, 16) or activation_bits not in (4, 8, 16):
            raise ValueError("SQ 仅支持已定义的 W4/W8/W16、A4/A8/A16 控制组")
        groups = apply_smoothquant_smoothing(
            model,
            entries,
            official,
            alpha=float(metadata.get("smooth_alpha", 0.5)),
        )
        for index, name in enumerate(names):
            module = modules[name]
            if weight_bits < 16:
                smoothquant_weight(module, official, bits=weight_bits)
            if activation_bits == 16:
                print(f"[official-smoothquant] activation fake quant disabled for {name}")
                continue
            handles.append(
                module.register_forward_pre_hook(
                    make_smoothquant_activation_hook(
                        module,
                        official["smooth_activation_quantize"],
                        bits=activation_bits,
                    )
                )
            )
            print(f"[official-smoothquant] {index + 1}/{len(names)} {name}")
        print(f"[official-smoothquant] smoothed norm/linear groups={groups}")
    return handles


def main() -> None:
    args = parse_args()
    from qvla.runtime_contract import assert_oft_runtime
    assert_oft_runtime()
    seed_all(args.seed, tensorflow=True)
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if args.libero_root and args.libero_root not in sys.path:
        sys.path.insert(0, args.libero_root)

    import experiments.robot.libero.run_libero_eval as L
    import experiments.robot.robot_utils as R
    from experiments.robot.libero.run_libero_eval import eval_libero
    original_load_initial_states = L.load_initial_states
    if args.a1_reset_dir is not None:
        def load_a1_initial_states(cfg, task_suite, task_id, log_file=None):
            if cfg.initial_states_path != "DEFAULT":
                raise ValueError("A1 reset directory cannot be combined with custom demo JSON")
            path = args.a1_reset_dir / f"task-{task_id:02d}.npy"
            states = np.load(path, allow_pickle=False)
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            L.log_message("A1_RESET_ARRAY " + json.dumps({"task_id": task_id,
                "path": str(path), "sha256": file_hash, "shape": list(states.shape),
                "dtype": str(states.dtype)}, sort_keys=True), log_file)
            return states, None
        L.load_initial_states = load_a1_initial_states
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    official = load_official_functions(args.official_root)
    entries, metadata = load_profiles(args.profile, args.method)
    awq_plan = None
    profile_manifest = [
        {"path": str(path), "sha256": source_sha256(path)} for path in args.profile
    ]
    scale_peft_state = None
    recovery_lora_state = None
    if args.awq_scale_peft_state is not None:
        scale_peft_state = load_scale_peft_state(args.awq_scale_peft_state)
        profile_manifest.append({
            "path": str(args.awq_scale_peft_state),
            "sha256": source_sha256(args.awq_scale_peft_state),
            "kind": "fixed_code_scale_peft_state",
            "parameter_tensors": len(scale_peft_state),
            "parameters": sum(value.numel() for value in scale_peft_state.values()),
            "serialized_bytes": args.awq_scale_peft_state.stat().st_size,
        })
    if args.awq_recovery_lora_state is not None:
        from qvla.model_registry import ADAPTER_LAYERS, validate_adapter_shapes
        if args.model_id == "B4":
            from qvla.b4_joint_peft import load_state, validate_artifact
            record = validate_artifact(args)
            recovery_lora_state = load_state(args.awq_recovery_lora_state, "B4", record["targets"])
        else:
            recovery_lora_state = load_recovery_lora_state(
                args.awq_recovery_lora_state, ADAPTER_LAYERS.get("B1" if args.model_id == "B2" else args.model_id))
        if args.model_id in ADAPTER_LAYERS:
            validate_adapter_shapes({k: tuple(v.shape) for k, v in recovery_lora_state.items()}, args.model_id)
        profile_manifest.append({
            "path": str(args.awq_recovery_lora_state),
            "sha256": source_sha256(args.awq_recovery_lora_state),
            "kind": "recovery_lora_state",
            "parameter_tensors": len(recovery_lora_state),
            "parameters": sum(value.numel() for value in recovery_lora_state.values()),
            "serialized_bytes": args.awq_recovery_lora_state.stat().st_size,
        })
    if args.model_id == "B2":
        from qvla.b4_joint_peft import load_state, validate_artifact
        record = validate_artifact(args)
        visual = load_state(args.awq_visual_lora_state, "B2", record["targets"])
        if set(visual) & set(recovery_lora_state):
            raise ValueError("Language/visual adapter keys overlap")
        recovery_lora_state.update(visual)
        profile_manifest.append({"path": str(args.awq_visual_lora_state),
            "sha256": source_sha256(args.awq_visual_lora_state), "kind": "B2_visual_increment",
            "parameter_tensors": len(visual), "parameters": sum(v.numel() for v in visual.values()),
            "serialized_bytes": args.awq_visual_lora_state.stat().st_size})
    if args.awq_candidate == "w2-no-clip-primary-g64":
        if args.method != "awq" or args.weight_bits != 2 or args.activation_bits != 16 or len(args.profile) != 1:
            raise ValueError("w2-no-clip-primary-g64 requires one AWQ W2A16 base profile")
        if args.awq_primary_group64_profile is None:
            raise ValueError("w2-no-clip-primary-g64 requires --awq-primary-group64-profile")
        from diagnostics.awq_interventions import current_w2_candidate_plan
        peer = load_profiles([args.awq_primary_group64_profile], "awq")
        awq_plan = current_w2_candidate_plan(entries, metadata, peer)
        profile_manifest.append({
            "path": str(args.awq_primary_group64_profile),
            "sha256": source_sha256(args.awq_primary_group64_profile),
        })
    elif args.awq_candidate == "w2-attention-no-clip-primary-g64":
        if args.method != "awq" or args.weight_bits != 2 or args.activation_bits != 16 or len(args.profile) != 1:
            raise ValueError("w2-attention-no-clip-primary-g64 requires one AWQ W2A16 base profile")
        from diagnostics.awq_interventions import attention_no_clip_primary_g64_plan
        peer = load_profiles([args.awq_primary_group64_profile], "awq")
        awq_plan = attention_no_clip_primary_g64_plan(entries, metadata, peer)
        profile_manifest.append({
            "path": str(args.awq_primary_group64_profile),
            "sha256": source_sha256(args.awq_primary_group64_profile),
        })
    elif args.awq_candidate in {"w2-attention-no-clip-dino-g128-siglip-g64", "w2-attention-no-clip-visual-g128"}:
        if args.method != "awq" or args.weight_bits != 2 or args.activation_bits != 16 or len(args.profile) != 1:
            raise ValueError("visual-group attribution requires one AWQ W2A16 base profile")
        from diagnostics.awq_interventions import attention_no_clip_visual_groups_plan
        peer = load_profiles([args.awq_primary_group64_profile], "awq")
        groups = {
            "w2-attention-no-clip-dino-g128-siglip-g64": (128, 64),
            "w2-attention-no-clip-visual-g128": (128, 128),
        }[args.awq_candidate]
        awq_plan = attention_no_clip_visual_groups_plan(entries, metadata, peer, *groups)
        profile_manifest.append({
            "path": str(args.awq_primary_group64_profile),
            "sha256": source_sha256(args.awq_primary_group64_profile),
        })
    elif args.awq_candidate == "w2-attention-primary-g64-stage-w4":
        from diagnostics.awq_interventions import attention_visual_stage_plan, parse_layers
        g64_peer = load_profiles([args.awq_primary_group64_profile], "awq")
        w4_peer = load_profiles([args.awq_w4_profile], "awq")
        awq_plan = attention_visual_stage_plan(
            entries, metadata, g64_peer, w4_peer, parse_layers(args.awq_w4_layers)
        )
        for path in (args.awq_primary_group64_profile, args.awq_w4_profile):
            profile_manifest.append({"path": str(path), "sha256": source_sha256(path)})
    elif args.awq_candidate == "w2-no-clip-stage-w4":
        from diagnostics.awq_interventions import language_stage_rollout_plan, parse_layers
        peer = load_profiles([args.awq_w4_profile], "awq")
        awq_plan = language_stage_rollout_plan(entries, metadata, parse_layers(args.awq_w4_layers), peer)
        profile_manifest.append({"path": str(args.awq_w4_profile), "sha256": source_sha256(args.awq_w4_profile)})
    elif args.awq_candidate == "w2-fixed-coordinates-stage-w4":
        from diagnostics.awq_interventions import family_precision_plan, vision_plan, parse_layers
        scales, targets, removed = family_precision_plan(entries, metadata, parse_layers(args.awq_w4_layers), "all")
        targets.update(vision_plan(entries, metadata, 2))
        awq_plan = scales, targets, removed
    from qvla.b4_joint_peft import validate_effective_plan
    effective_targets = (awq_plan[1] if awq_plan is not None else
                         {name: (entries[name], args.weight_bits, metadata.get("group_size", 128))
                          for name in select_awq_scope(list(entries), args.awq_scope)})
    validate_effective_plan(args.model_id, effective_targets)
    if checkpoint_identity(args.pretrained_checkpoint) != metadata["checkpoint_identity"]:
        raise RuntimeError("Profile 检查点内容指纹不符（允许迁移路径，不允许换权重）")
    if args.model_id in ("B2", "B4"):
        from qvla.b4_joint_peft import validate_artifact
        if validate_artifact(args)["checkpoint_identity"] != metadata["checkpoint_identity"]:
            raise ValueError("Candidate/checkpoint identity mismatch")
    current_sources = {
        "awq_auto_scale": official["awq_parent"] / "awq/quantize/auto_scale.py",
        "awq_auto_clip": official["awq_parent"] / "awq/quantize/auto_clip.py",
        "awq_quantizer": official["awq_parent"] / "awq/quantize/quantizer.py",
        "smoothquant_smooth": official["smoothquant_parent"] / "smoothquant/smooth.py",
        "smoothquant_fake_quant": official["smoothquant_parent"] / "smoothquant/fake_quant.py",
    }
    for source_name, source_path in current_sources.items():
        expected = metadata.get("official_sources", {}).get(source_name, {}).get("sha256")
        if not expected or source_sha256(source_path) != expected:
            raise RuntimeError(f"Official source hash mismatch: {source_name}")
    counts = scope_counts(list(entries))
    print("[official-quant] scope=" + json.dumps(counts, sort_keys=True))
    if counts != {
        "total": 422,
        "primary_vision": 93,
        "fused_vision": 105,
        "language": 224,
        "vision": 198,
        "unexpected": 0,
    }:
        raise RuntimeError(f"Expected exact QVLA connected scope, got {counts}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OpenVLAForActionPrediction.from_pretrained(
        args.pretrained_checkpoint,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation=metadata["llm_attention"],
    ).to(device)
    model.eval()
    if source_sha256(Path(inspect.getsourcefile(type(model)))) != metadata["model_source_sha256"]:
        raise RuntimeError("校准/评估实际模型类源码不一致")
    model.language_model.config.use_cache = False
    model.vision_backbone.set_num_images_in_input(2)
    statistics = Path(args.pretrained_checkpoint) / "dataset_statistics.json"
    if statistics.is_file():
        model.norm_stats = json.loads(statistics.read_text())

    handles = apply_quantization(
        model,
        entries,
        metadata,
        args.method,
        official,
        args.weight_bits,
        args.activation_bits,
        awq_plan=awq_plan,
        awq_scope=args.awq_scope,
        scale_peft_state=scale_peft_state,
        recovery_lora_state=recovery_lora_state,
    )
    print(
        f"[official-quant] method={args.method} weight_bits={args.weight_bits} "
        f"activation_bits={args.activation_bits} profile_targets={len(entries)} scope={args.awq_scope}"
    )
    print("[official-quant] BF16 exclusions=projector,proprio_projector,action_head,embeddings,norms")
    print("[official-quant] candidate=" + json.dumps({
        "model_id": args.model_id,
        "identity_status": "canonical" if args.model_id else "archival_or_diagnostic_unregistered",
        "name": args.awq_candidate,
        "scope": args.awq_scope,
        "applied_targets": len(select_awq_scope(list(entries), args.awq_scope)) if args.method == "awq" else len(entries),
        "profiles": profile_manifest,
        "w4_layers": args.awq_w4_layers,
        "applied_target_bits": (
            {str(bits): sum(1 for name in select_awq_scope(list(entries), args.awq_scope)
                           if awq_plan[1][name][1] == bits)
             for bits in sorted({value[1] for value in awq_plan[1].values()})}
            if awq_plan is not None else {str(args.weight_bits): len(select_awq_scope(list(entries), args.awq_scope))}
        ) if args.method == "awq" else {},
        "removed_language_clips": len(awq_plan[2]) if awq_plan is not None else 0,
        "target_group_counts": (
            {str(group): sum(1 for _entry, _bits, value_group in awq_plan[1].values() if value_group == group)
             for group in sorted({value_group for _entry, _bits, value_group in awq_plan[1].values()})}
            if awq_plan is not None else {str(metadata.get("group_size", 128)): len(entries)}
        ),
    }, sort_keys=True))

    original_get_model = R.get_model
    original_libero_get_model = getattr(L, "get_model", None)

    def patched_get_model(_cfg: Any):
        print("[official-quant] using preloaded model")
        return model

    R.get_model = patched_get_model
    if original_libero_get_model is not None:
        L.get_model = patched_get_model
    argv_backup = list(sys.argv)
    original_get_action = L.get_action
    original_run_episode = L.run_episode
    trace_file = None
    observation_capture = None
    h17_recorder = None
    if args.trace_actions or args.trace_observations:
        import numpy as np
        os.makedirs(args.local_log_dir, exist_ok=True)
        if args.trace_actions:
            trace_file = open(Path(args.local_log_dir) / "policy-queries.jsonl", "x", encoding="utf-8")
        if args.trace_observations:
            from prismatic.vla.constants import NUM_ACTIONS_CHUNK
            from qvla.on_policy_capture import H17SummaryRecorder, OnPolicyObservationCapture
            observation_capture = OnPolicyObservationCapture(Path(args.local_log_dir), NUM_ACTIONS_CHUNK)
            h17_recorder = H17SummaryRecorder(model)

            def traced_run_episode(*run_args, **run_kwargs):
                task = run_kwargs.get("task_description")
                if task is None and len(run_args) > 2:
                    task = run_args[2]
                if not isinstance(task, str):
                    raise RuntimeError("Cannot identify task for P2.5 observation capture")
                observation_capture.begin_episode(task)
                try:
                    result = original_run_episode(*run_args, **run_kwargs)
                except BaseException:
                    observation_capture.end_episode(False, aborted=True)
                    raise
                observation_capture.end_episode(bool(result[0]))
                return result

            L.run_episode = traced_run_episode

        def traced_get_action(cfg, model, obs, task_label, **kwargs):
            state_before_action = np.asarray(obs["state"]).copy()
            if h17_recorder is not None:
                h17_recorder.reset()
            actions = original_get_action(cfg, model, obs, task_label, **kwargs)
            chunk = np.asarray(actions).copy()
            if trace_file is not None:
                trace_file.write(json.dumps({
                    "task": task_label, "state": state_before_action.tolist(),
                    "state_space": "raw_proprio_before_get_action",
                    "raw_policy_chunk": chunk.tolist(), "finite": bool(np.isfinite(chunk).all()),
                    "note": "query-time observation; raw chunk before gripper processing; later closed-loop states diverge",
                }) + "\n")
                trace_file.flush()
            if observation_capture is not None:
                observation_capture.record_query(obs, task_label, chunk, h17_recorder.take())
            return actions

        L.get_action = traced_get_action
    try:
        sys.argv = [
            sys.argv[0],
            "--model_family", "openvla",
            "--pretrained_checkpoint", args.pretrained_checkpoint,
            "--task_suite_name", args.task_suite_name,
            "--num_trials_per_task", str(args.num_trials_per_task),
            "--initial_state_offset", str(args.initial_state_offset),
            "--local_log_dir", args.local_log_dir,
            "--center_crop", "True",
            "--seed", str(args.seed),
            "--env_seed", str(args.env_seed),
            "--seed_protocol", args.seed_protocol,
            "--attn_implementation", metadata["llm_attention"],
        ]
        os.makedirs(args.local_log_dir, exist_ok=True)
        eval_libero()
    finally:
        L.get_action = original_get_action
        L.run_episode = original_run_episode
        L.load_initial_states = original_load_initial_states
        if trace_file is not None:
            trace_file.close()
        if h17_recorder is not None:
            h17_recorder.close()
        if observation_capture is not None:
            observation_capture.close()
        sys.argv = argv_backup
        R.get_model = original_get_model
        if original_libero_get_model is not None:
            L.get_model = original_libero_get_model
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()
