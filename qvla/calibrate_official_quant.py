from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qvla.action_jacobian_batch import initialize, load_sample, prepare_inputs
from qvla.baseline_contract import CONTRACT_ID, check_scope, checkpoint_identity, sample_identity
from qvla.reproducibility import seed_all
from qvla.official_quant_adapter import (
    AWQ_GROUP_SIZE,
    PROFILE_FORMAT,
    calibrate_awq_entry,
    input_absmax,
    load_official_functions,
    module_rows,
    read_targets,
    scope_counts,
    source_sha256,
    validate_targets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate official AWQ W4A16 or SmoothQuant W4A4 for OpenVLA."
    )
    parser.add_argument("--method", required=True, choices=("awq", "smoothquant"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples-dir", required=True, type=Path)
    parser.add_argument("--targets-file", required=True, type=Path)
    parser.add_argument("--official-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--bits", type=int, default=4, choices=(2, 4, 8))
    parser.add_argument("--awq-search", choices=("official-block", "linear-diagnostic"), default="official-block")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-rows", type=int, default=256)
    parser.add_argument("--max-output-rows", type=int, default=256)
    parser.add_argument("--clip-tokens", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=AWQ_GROUP_SIZE)
    parser.add_argument("--smooth-alpha", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(args.seed, tensorflow=True)
    if args.output.exists() or args.summary_json.exists():
        raise FileExistsError("拒绝覆盖已有 profile / 摘要，请换一个输出目录")
    if min(args.num_samples, args.max_rows, args.max_output_rows, args.clip_tokens, args.group_size) < 1:
        raise ValueError("样本数/行预算/group size 必须为正数")
    if not 0 <= args.smooth_alpha <= 1:
        raise ValueError("alpha 必须位于 [0,1]")
    if args.method == "smoothquant" and args.bits != 4:
        raise ValueError("SQ 校准存储 absmax，以 W4 为基准；评估可做 W8/W16 控制")
    official = load_official_functions(args.official_root)
    names = read_targets(args.targets_file)
    check_scope(names, complete=args.method == "awq" and args.awq_search == "official-block")
    counts = scope_counts(names)
    if counts["unexpected"]:
        raise RuntimeError(f"Unexpected target scope: {counts}")

    samples = sorted(args.samples_dir.glob("sample-*.npz"))[: args.num_samples]
    if len(samples) != args.num_samples:
        raise RuntimeError(f"Requested {args.num_samples} samples, found {len(samples)}")

    cfg, model, action_head, proprio_projector, processor = initialize(
        args.checkpoint, args.seed
    )
    # 固定无 cache 的真实 VLA 前向，官方 block 搜索使用相同 backend。
    model.language_model.config.use_cache = False
    identity = checkpoint_identity(args.checkpoint)
    from qvla.awq_block import tensor_tree, calibrate_llama_block
    block_records = []
    use_blocks = args.method == "awq" and args.awq_search == "official-block"
    selected = validate_targets(model, names)
    device = next(model.parameters()).device
    # 每次模型调用先取固定配额，最后再统一下采样。视觉层每个样本会由
    # 主相机和腕部相机各调用一次，因此不能在前 16 个样本就耗尽全局预算。
    if args.method == "awq" and args.max_rows < args.num_samples * 2:
        raise ValueError("AWQ 行预算至少覆盖每帧两个相机，max_rows >= 2*num_samples")
    rows: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    capture_calls = {name: 0 for name in names}
    maxima: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        module = selected[name]

        def hook(_module: torch.nn.Module, values: tuple[Any, ...], _output: Any):
            value = values[0]
            capture_calls[name] += 1
            current_max = input_absmax(module, value).cpu()
            maxima[name] = (
                current_max
                if name not in maxima
                else torch.maximum(maxima[name], current_max)
            )
            if args.method == "awq" and not (use_blocks and name.startswith("language_model.")):
                current = module_rows(module, value)
                calls_per_sample = 2 if name.startswith("vision_backbone.") else 1
                take = min(args.max_rows // (args.num_samples * calls_per_sample), current.shape[0])
                if take:
                    index = torch.linspace(0, current.shape[0] - 1, take, device=current.device)
                    chosen = current[index.round().long()].to(torch.bfloat16).cpu()
                    rows[name].append(chosen)

        return hook

    for name, module in selected.items():
        handles.append(module.register_forward_hook(make_hook(name)))
    if use_blocks:
        def capture_first(_module, values, kwargs):
            hidden = values[0] if values else kwargs["hidden_states"]
            context = {k: v for k, v in kwargs.items() if k != "hidden_states"}
            block_records.append((tensor_tree(hidden, "cpu"), tensor_tree(context, "cpu")))
        handles.append(model.language_model.model.layers[0].register_forward_pre_hook(capture_first, with_kwargs=True))
    try:
        for index, path in enumerate(samples):
            calls_before = dict(capture_calls)
            sample = load_sample(path)
            model_inputs, state = prepare_inputs(sample, cfg, model, processor, device)
            with torch.no_grad():
                model.predict_action(
                    **model_inputs,
                    unnorm_key=cfg.unnorm_key,
                    do_sample=False,
                    proprio=state,
                    proprio_projector=proprio_projector,
                    action_head=action_head,
                    noisy_action_projector=None,
                    use_film=False,
                )
            for name in names:
                expected_calls = 2 if name.startswith("vision_backbone.") else 1
                if capture_calls[name] - calls_before[name] != expected_calls:
                    raise RuntimeError(f"相机/分支调用数异常：{path.name} {name}")
            print(f"[capture] {index + 1}/{len(samples)} {path.name}")
    finally:
        for handle in handles:
            handle.remove()

    missing = [name for name in names if name not in maxima]
    if missing:
        raise RuntimeError(f"Targets were not executed: {missing}")

    payload: dict[str, Any] = {
        "format_version": PROFILE_FORMAT,
        "contract_id": CONTRACT_ID,
        "implementation_sources": {name: source_sha256(Path(__file__).with_name(name)) for name in
            ("official_quant_adapter.py", "awq_block.py", "action_jacobian_batch.py", "runtime_contract.py")},
        "model_source_sha256": source_sha256(Path(inspect.getsourcefile(type(model)))),
        "checkpoint_identity": identity,
        "sample_sha256": sample_identity(samples),
        "seed": args.seed,
        "calibration_manifest": json.loads((args.samples_dir / "manifest.json").read_text(encoding="utf-8"))
            if (args.samples_dir / "manifest.json").exists() else None,
        "llm_attention": model.language_model.config._attn_implementation,
        "algorithm": "awq_official_llama_vision_adapter_v1" if use_blocks else
                     ("linear_diagnostic_NOT_official_block" if args.method == "awq" else "sq_official_primitives_vit_adapter_v1"),
        "block_scales": {},
        "method": args.method,
        "bits": args.bits,
        "activation_bits": 16 if args.method == "awq" else 4,
        "checkpoint": str(args.checkpoint),
        "num_samples": args.num_samples,
        "sample_names": [path.name for path in samples],
        "scope": counts,
        "targets": names,
        "entries": {},
        "official_sources": {
            "awq_auto_scale": {
                "sha256": source_sha256(official["awq_parent"] / "awq/quantize/auto_scale.py"),
            },
            "awq_auto_clip": {
                "sha256": source_sha256(official["awq_parent"] / "awq/quantize/auto_clip.py"),
            },
            "awq_quantizer": {
                "path": str(official["awq_parent"] / "awq/quantize/quantizer.py"),
                "sha256": source_sha256(official["awq_parent"] / "awq/quantize/quantizer.py"),
            },
            "smoothquant_smooth": {
                "path": str(official["smoothquant_parent"] / "smoothquant/smooth.py"),
                "sha256": source_sha256(official["smoothquant_parent"] / "smoothquant/smooth.py"),
            },
            "smoothquant_fake_quant": {
                "path": str(official["smoothquant_parent"] / "smoothquant/fake_quant.py"),
                "sha256": source_sha256(official["smoothquant_parent"] / "smoothquant/fake_quant.py"),
            },
        },
        "adapter_notes": {
            "openvla_scope": "422 causally connected vision/language targets",
            "conv2d": "Patch embedding Conv2d is flattened for weight quantization",
            "awq": "Llama 官方 block search；ViT/Conv 明确为逐 Linear 适配；非官方 VLA recipe",
            "smoothquant": "Official smoothing and fake-quant primitives with n_bits=4",
        },
    }

    errors = []
    for index, name in enumerate(names):
        if use_blocks and name.startswith("language_model."):
            continue
        module = selected[name]
        if args.method == "smoothquant":
            entry = {
                "module_type": type(module).__name__,
                "shape": list(module.weight.shape),
                "activation_absmax": maxima[name].float(),
                "capture_calls": capture_calls[name],
            }
        else:
            calibration_rows = torch.cat(rows[name], dim=0)
            if calibration_rows.shape[0] > args.max_rows:
                index_rows = torch.linspace(
                    0, calibration_rows.shape[0] - 1, args.max_rows
                ).round().long().unique()
                calibration_rows = calibration_rows[index_rows]
            calibration_rows = calibration_rows.float().to(device)
            entry = calibrate_awq_entry(
                name,
                module,
                calibration_rows,
                official,
                bits=args.bits,
                group_size=args.group_size,
                max_output_rows=args.max_output_rows,
                clip_tokens=args.clip_tokens,
            )
            errors.append(entry["relative_mse"])
            entry["capture_calls"] = capture_calls[name]
            entry["calibration_rows"] = calibration_rows.shape[0]
            del calibration_rows
            torch.cuda.empty_cache()
        payload["entries"][name] = entry
        print(f"[target] {index + 1}/{len(names)} {name}")

    if use_blocks:
        if len(block_records) != len(samples):
            raise RuntimeError("首个 Llama block 校准帧数不匹配")
        payload["block_equivalence"] = {}
        for i, block in enumerate(model.language_model.model.layers):
            prefix = f"language_model.model.layers.{i}"
            block_entries, scales, block_records, equality = calibrate_llama_block(
                block, block_records, official, args.bits, args.group_size, args.clip_tokens)
            payload["entries"].update({prefix + "." + n: e for n, e in block_entries.items()})
            payload["block_scales"][prefix] = scales
            payload["block_equivalence"][prefix] = equality
            print(f"[official-block] {i + 1}/32 max_smooth_rmse2={max(equality):.6g}", flush=True)
            torch.cuda.empty_cache()
        check_scope(payload["entries"])

    payload["group_size"] = args.group_size
    payload["smooth_alpha"] = args.smooth_alpha
    if errors:
        payload["mean_relative_mse"] = sum(errors) / len(errors)

    summary = {
        "format_version": PROFILE_FORMAT,
        "method": args.method,
        "weight_bits": args.bits,
        "activation_bits": payload["activation_bits"],
        "num_samples": args.num_samples,
        "scope": counts,
        "entries": len(payload["entries"]),
        "mean_relative_mse": payload.get("mean_relative_mse"),
        "output": str(args.output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    args.summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("OFFICIAL QUANTIZATION CALIBRATION: PASS")


if __name__ == "__main__":
    main()
