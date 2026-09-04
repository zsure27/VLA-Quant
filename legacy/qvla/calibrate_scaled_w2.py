from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from qvla.action_jacobian_batch import initialize, load_sample, prepare_inputs


DEFAULT_TARGETS = (
    "vision_backbone.featurizer.patch_embed.proj",
    "vision_backbone.featurizer.blocks.0.attn.proj",
    "vision_backbone.featurizer.blocks.1.attn.proj",
)


def parse_csv_floats(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate RTN, SmoothQuant-style, and AWQ-style W2A16 profiles."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--max-linear-rows", type=int, default=2048)
    parser.add_argument("--max-conv-batches", type=int, default=8)
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--targets",
        default=",".join(DEFAULT_TARGETS),
        help="comma-separated module names",
    )
    parser.add_argument(
        "--alpha-grid",
        type=parse_csv_floats,
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument(
        "--clip-grid",
        type=parse_csv_floats,
        default=(1.0, 0.95, 0.9, 0.85, 0.8),
    )
    return parser.parse_args()


def quantize_rows_symmetric(
    weight: torch.Tensor,
    bits: int,
    clip_ratio: float = 1.0,
) -> torch.Tensor:
    original_shape = weight.shape
    rows = weight.reshape(weight.shape[0], -1)
    qmax = (1 << (bits - 1)) - 1
    xmax = rows.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    xmax = xmax * clip_ratio
    scale = xmax.clamp_min(1e-8) / qmax
    quantized = torch.round(rows / scale).clamp(-(qmax + 1), qmax)
    return (quantized * scale).reshape(original_shape)


def flatten_linear_inputs(
    inputs: list[torch.Tensor],
    max_rows: int,
    seed: int,
) -> torch.Tensor:
    rows = torch.cat([value.reshape(-1, value.shape[-1]) for value in inputs], dim=0)
    if rows.shape[0] > max_rows:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        index = torch.randperm(rows.shape[0], generator=generator)[:max_rows]
        rows = rows[index]
    return rows.contiguous()


def select_conv_inputs(inputs: list[torch.Tensor], max_batches: int) -> torch.Tensor:
    selected: list[torch.Tensor] = []
    batches = 0
    for value in inputs:
        if batches >= max_batches:
            break
        remaining = max_batches - batches
        selected.append(value[:remaining])
        batches += min(value.shape[0], remaining)
    if not selected:
        raise RuntimeError("no Conv2d calibration input captured")
    return torch.cat(selected, dim=0).contiguous()


def input_channel_stat(module: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    if isinstance(module, torch.nn.Linear):
        return inputs.abs().amax(dim=0).clamp_min(1e-6)
    if isinstance(module, torch.nn.Conv2d):
        return inputs.abs().amax(dim=(0, 2, 3)).clamp_min(1e-6)
    raise TypeError(type(module))


def weight_channel_stat(
    module: torch.nn.Module,
    weight: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if isinstance(module, torch.nn.Linear):
        reduce_dims = (0,)
    elif isinstance(module, torch.nn.Conv2d):
        reduce_dims = (0, 2, 3)
    else:
        raise TypeError(type(module))

    if mode == "smoothquant":
        result = weight.abs().amax(dim=reduce_dims)
    elif mode == "awq":
        result = weight.abs().mean(dim=reduce_dims)
    else:
        raise ValueError(mode)
    return result.clamp_min(1e-6)


def make_input_scale(
    activation_stat: torch.Tensor,
    weight_stat: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    scale = activation_stat.pow(alpha) / weight_stat.pow(1.0 - alpha)
    scale = scale.clamp(1e-4, 1e4)
    normalizer = torch.sqrt(scale.max() * scale.min()).clamp_min(1e-8)
    return (scale / normalizer).clamp(1e-4, 1e4)


def transform_weight(
    module: torch.nn.Module,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
) -> torch.Tensor:
    if isinstance(module, torch.nn.Linear):
        return weight * input_scale.view(1, -1)
    if isinstance(module, torch.nn.Conv2d):
        return weight * input_scale.view(1, -1, 1, 1)
    raise TypeError(type(module))


def transform_input(
    module: torch.nn.Module,
    inputs: torch.Tensor,
    input_scale: torch.Tensor,
) -> torch.Tensor:
    if isinstance(module, torch.nn.Linear):
        return inputs / input_scale.view(1, -1)
    if isinstance(module, torch.nn.Conv2d):
        return inputs / input_scale.view(1, -1, 1, 1)
    raise TypeError(type(module))


def module_output(
    module: torch.nn.Module,
    inputs: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if isinstance(module, torch.nn.Linear):
        return F.linear(inputs, weight, bias=None)
    if isinstance(module, torch.nn.Conv2d):
        return F.conv2d(
            inputs,
            weight,
            bias=None,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )
    raise TypeError(type(module))


@torch.no_grad()
def score_candidate(
    module: torch.nn.Module,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    bits: int,
    clip_ratio: float,
    reference: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    scaled_weight = transform_weight(module, weight, input_scale)
    quantized_weight = quantize_rows_symmetric(scaled_weight, bits, clip_ratio)
    scaled_input = transform_input(module, inputs, input_scale)
    candidate = module_output(module, scaled_input, quantized_weight)
    mse = (candidate - reference).float().square().mean()
    denominator = reference.float().square().mean().clamp_min(1e-12)
    relative_mse = float((mse / denominator).item())
    return relative_mse, quantized_weight


def profile_entry(
    method: str,
    bits: int,
    alpha: float,
    clip_ratio: float,
    relative_mse: float,
    input_scale: torch.Tensor,
    quantized_weight: torch.Tensor,
    module: torch.nn.Module,
) -> dict[str, Any]:
    return {
        "method": method,
        "bits": bits,
        "alpha": alpha,
        "clip_ratio": clip_ratio,
        "relative_mse": relative_mse,
        "module_type": type(module).__name__,
        "input_scale": input_scale.detach().cpu().float(),
        "quantized_weight": quantized_weight.detach().cpu().to(torch.bfloat16),
    }


def main() -> None:
    args = parse_args()
    if args.bits != 2:
        raise ValueError("This controlled experiment is intentionally fixed to W2A16")
    if args.num_samples < 1:
        raise ValueError("num-samples must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")
    targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    if len(targets) != len(set(targets)):
        raise ValueError("duplicate target names")

    sample_paths = sorted(args.samples_dir.glob("sample-*.npz"))[: args.num_samples]
    if len(sample_paths) != args.num_samples:
        raise RuntimeError(
            f"requested {args.num_samples} samples, found {len(sample_paths)} in {args.samples_dir}"
        )

    cfg, model, action_head, proprio_projector, processor = initialize(
        args.checkpoint, args.seed
    )
    modules = dict(model.named_modules())
    for name in targets:
        if name not in modules:
            raise KeyError(f"target not found: {name}")
        if not isinstance(modules[name], (torch.nn.Linear, torch.nn.Conv2d)):
            raise TypeError(f"unsupported target {name}: {type(modules[name])}")

    captured: dict[str, list[torch.Tensor]] = {name: [] for name in targets}
    handles = []

    def make_hook(name: str):
        def hook(_module, inputs, _output):
            captured[name].append(inputs[0].detach().float().cpu())
        return hook

    for name in targets:
        handles.append(modules[name].register_forward_hook(make_hook(name)))

    try:
        for index, sample_path in enumerate(sample_paths):
            sample = load_sample(sample_path)
            model_inputs, state = prepare_inputs(
                sample, cfg, model, processor, device
            )
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
            print(f"[capture] {index + 1}/{len(sample_paths)} {sample_path.name}")
    finally:
        for handle in handles:
            handle.remove()

    payload: dict[str, Any] = {
        "format_version": 1,
        "checkpoint": str(args.checkpoint),
        "bits": args.bits,
        "num_samples": len(sample_paths),
        "sample_names": [path.name for path in sample_paths],
        "targets": targets,
        "profiles": {
            "rtn": {"targets": {}},
            "smoothquant": {"targets": {}},
            "awq": {"targets": {}},
        },
    }
    summary: dict[str, Any] = {
        "bits": args.bits,
        "num_samples": len(sample_paths),
        "targets": {},
    }

    for target_index, name in enumerate(targets):
        module = modules[name]
        if isinstance(module, torch.nn.Linear):
            calibration_input = flatten_linear_inputs(
                captured[name], args.max_linear_rows, args.seed + target_index
            )
        else:
            calibration_input = select_conv_inputs(
                captured[name], args.max_conv_batches
            )

        calibration_input = calibration_input.to(device=device, dtype=torch.float32)
        weight = module.weight.detach().to(device=device, dtype=torch.float32)
        reference = module_output(module, calibration_input, weight)
        activation_stat = input_channel_stat(module, calibration_input)

        ones = torch.ones_like(activation_stat)
        rtn_mse, rtn_weight = score_candidate(
            module, calibration_input, weight, ones, args.bits, 1.0, reference
        )
        payload["profiles"]["rtn"]["targets"][name] = profile_entry(
            "rtn", args.bits, 0.0, 1.0, rtn_mse, ones, rtn_weight, module
        )

        best_smooth: tuple[float, float, torch.Tensor, torch.Tensor] | None = None
        smooth_weight_stat = weight_channel_stat(module, weight, "smoothquant")
        for alpha in args.alpha_grid:
            scale = make_input_scale(activation_stat, smooth_weight_stat, alpha)
            mse, quantized_weight = score_candidate(
                module, calibration_input, weight, scale, args.bits, 1.0, reference
            )
            if best_smooth is None or mse < best_smooth[0]:
                best_smooth = (mse, alpha, scale, quantized_weight)
        assert best_smooth is not None
        payload["profiles"]["smoothquant"]["targets"][name] = profile_entry(
            "smoothquant",
            args.bits,
            best_smooth[1],
            1.0,
            best_smooth[0],
            best_smooth[2],
            best_smooth[3],
            module,
        )

        best_awq: tuple[float, float, float, torch.Tensor, torch.Tensor] | None = None
        awq_weight_stat = weight_channel_stat(module, weight, "awq")
        for alpha in args.alpha_grid:
            scale = make_input_scale(activation_stat, awq_weight_stat, alpha)
            for clip_ratio in args.clip_grid:
                mse, quantized_weight = score_candidate(
                    module,
                    calibration_input,
                    weight,
                    scale,
                    args.bits,
                    clip_ratio,
                    reference,
                )
                if best_awq is None or mse < best_awq[0]:
                    best_awq = (
                        mse,
                        alpha,
                        clip_ratio,
                        scale,
                        quantized_weight,
                    )
        assert best_awq is not None
        payload["profiles"]["awq"]["targets"][name] = profile_entry(
            "awq",
            args.bits,
            best_awq[1],
            best_awq[2],
            best_awq[0],
            best_awq[3],
            best_awq[4],
            module,
        )

        target_summary = {}
        for method in ("rtn", "smoothquant", "awq"):
            entry = payload["profiles"][method]["targets"][name]
            target_summary[method] = {
                "alpha": entry["alpha"],
                "clip_ratio": entry["clip_ratio"],
                "relative_mse": entry["relative_mse"],
            }
        summary["targets"][name] = target_summary

        print(f"\n[target] {name}")
        for method, values in target_summary.items():
            print(method, values)

        del calibration_input, weight, reference, activation_stat
        torch.cuda.empty_cache()

    for method in ("rtn", "smoothquant", "awq"):
        values = [
            payload["profiles"][method]["targets"][name]["relative_mse"]
            for name in targets
        ]
        payload["profiles"][method]["mean_relative_mse"] = sum(values) / len(values)
        summary[method] = {
            "mean_relative_mse": sum(values) / len(values)
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    args.summary_json.write_text(json.dumps(summary, indent=2))

    print("\nsummary:", args.summary_json)
    print("calibration:", args.output)
    print("SCALED W2 CALIBRATION: PASS")


if __name__ == "__main__":
    main()
