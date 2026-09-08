from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

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
    parser.add_argument("--local_log_dir", required=True)
    parser.add_argument("--libero_root", default=os.environ.get("LIBERO_ROOT", ""))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument("--seed-protocol", choices=("upstream", "paired"), default="upstream")
    return parser.parse_args()


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


@torch.no_grad()
def apply_quantization(
    model: torch.nn.Module,
    entries: dict[str, Any],
    metadata: dict[str, Any],
    method: str,
    official: dict[str, Any],
    weight_bits: int,
    activation_bits: int,
) -> list[Any]:
    names = list(entries)
    check_scope(names)
    modules = validate_targets(model, names)
    for name, module in modules.items():
        entry = entries[name]
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
        if set(metadata.get("block_scales", {})) != expected_blocks:
            raise ValueError("官方 block scale 不完整，禁止隐式退化到 RTN")
        for prefix, scales in metadata["block_scales"].items():
            apply_block_scales(all_modules[prefix], scales, official)
        for index, name in enumerate(names):
            apply_entry = apply_llama_entry if name.startswith("language_model.") else apply_awq_entry
            if name.startswith("language_model.") and entries[name].get("recipe") != "official_llama_block_v1":
                raise ValueError("LLM profile 不是官方 block 搜索")
            apply_entry(
                modules[name],
                entries[name],
                official,
                bits=weight_bits,
                group_size=int(metadata.get("group_size", 128)),
            )
            print(f"[official-awq] {index + 1}/{len(names)} {name}")
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
    if checkpoint_identity(args.pretrained_checkpoint) != metadata["checkpoint_identity"]:
        raise RuntimeError("Profile 检查点内容指纹不符（允许迁移路径，不允许换权重）")
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
    )
    print(
        f"[official-quant] method={args.method} weight_bits={args.weight_bits} "
        f"activation_bits={args.activation_bits} targets={len(entries)}"
    )
    print("[official-quant] BF16 exclusions=projector,proprio_projector,action_head,embeddings,norms")

    original_get_model = R.get_model
    original_libero_get_model = getattr(L, "get_model", None)

    def patched_get_model(_cfg: Any):
        print("[official-quant] using preloaded model")
        return model

    R.get_model = patched_get_model
    if original_libero_get_model is not None:
        L.get_model = patched_get_model
    argv_backup = list(sys.argv)
    try:
        sys.argv = [
            sys.argv[0],
            "--model_family", "openvla",
            "--pretrained_checkpoint", args.pretrained_checkpoint,
            "--task_suite_name", args.task_suite_name,
            "--num_trials_per_task", str(args.num_trials_per_task),
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
        sys.argv = argv_backup
        R.get_model = original_get_model
        if original_libero_get_model is not None:
            L.get_model = original_libero_get_model
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()
