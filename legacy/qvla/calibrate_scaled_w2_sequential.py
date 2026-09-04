from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qvla.action_jacobian_batch import initialize, load_sample, prepare_inputs
from qvla.calibrate_scaled_w2 import (
    flatten_linear_inputs,
    input_channel_stat,
    make_input_scale,
    module_output,
    parse_csv_floats,
    profile_entry,
    score_candidate,
    select_conv_inputs,
    weight_channel_stat,
)
from qvla.run_eval_scaled_w2 import make_input_scale_hook


class StopAtTarget(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially calibrate an AWQ-style W2A16 profile. Each target is "
            "applied before calibration continues to downstream targets."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--max-linear-rows", type=int, default=2048)
    parser.add_argument("--max-conv-batches", type=int, default=8)
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
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


def run_model(
    model: Any,
    action_head: Any,
    proprio_projector: Any,
    cfg: Any,
    model_inputs: dict[str, torch.Tensor],
    state: np.ndarray,
) -> None:
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


@torch.no_grad()
def capture_target_inputs(
    target: torch.nn.Module,
    prepared_samples: list[tuple[dict[str, torch.Tensor], np.ndarray]],
    model: Any,
    action_head: Any,
    proprio_projector: Any,
    cfg: Any,
) -> list[torch.Tensor]:
    captured: list[torch.Tensor] = []

    def capture_and_stop(_module, args):
        if not args:
            raise RuntimeError("target received no positional input")
        captured.append(args[0].detach().float().cpu())
        raise StopAtTarget

    handle = target.register_forward_pre_hook(capture_and_stop)
    try:
        for model_inputs, state in prepared_samples:
            try:
                run_model(
                    model,
                    action_head,
                    proprio_projector,
                    cfg,
                    model_inputs,
                    state,
                )
            except StopAtTarget:
                continue
            raise RuntimeError("target was not executed during model forward")
    finally:
        handle.remove()

    if len(captured) != len(prepared_samples):
        raise RuntimeError(
            f"captured {len(captured)} inputs for {len(prepared_samples)} samples"
        )
    return captured


@torch.no_grad()
def search_awq(
    module: torch.nn.Module,
    calibration_input: torch.Tensor,
    bits: int,
    alpha_grid: tuple[float, ...],
    clip_grid: tuple[float, ...],
) -> dict[str, Any]:
    weight = module.weight.detach().to(
        device=calibration_input.device,
        dtype=torch.float32,
    )
    reference = module_output(module, calibration_input, weight)
    activation_stat = input_channel_stat(module, calibration_input)
    weight_stat = weight_channel_stat(module, weight, "awq")

    best: tuple[float, float, float, torch.Tensor, torch.Tensor] | None = None
    for alpha in alpha_grid:
        input_scale = make_input_scale(activation_stat, weight_stat, alpha)
        for clip_ratio in clip_grid:
            mse, quantized_weight = score_candidate(
                module,
                calibration_input,
                weight,
                input_scale,
                bits,
                clip_ratio,
                reference,
            )
            if best is None or mse < best[0]:
                best = (
                    mse,
                    alpha,
                    clip_ratio,
                    input_scale,
                    quantized_weight,
                )

    assert best is not None
    return profile_entry(
        "awq",
        bits,
        best[1],
        best[2],
        best[0],
        best[3],
        best[4],
        module,
    )


@torch.no_grad()
def apply_entry(
    module: torch.nn.Module,
    entry: dict[str, Any],
) -> Any:
    module.weight.copy_(
        entry["quantized_weight"].to(
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
    )
    return module.register_forward_pre_hook(
        make_input_scale_hook(module, entry["input_scale"].float())
    )


def main() -> None:
    args = parse_args()
    if args.bits != 2:
        raise ValueError("This controlled experiment is fixed to W2A16")
    if args.num_samples < 1:
        raise ValueError("num-samples must be positive")

    targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    if not targets or len(targets) != len(set(targets)):
        raise ValueError("targets must be a non-empty list without duplicates")

    sample_paths = sorted(args.samples_dir.glob("sample-*.npz"))[: args.num_samples]
    if len(sample_paths) != args.num_samples:
        raise RuntimeError(
            f"requested {args.num_samples} samples, found {len(sample_paths)}"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")
    cfg, model, action_head, proprio_projector, processor = initialize(
        args.checkpoint,
        args.seed,
    )
    modules = dict(model.named_modules())
    for name in targets:
        if name not in modules:
            raise KeyError(f"target not found: {name}")
        if not isinstance(modules[name], (torch.nn.Linear, torch.nn.Conv2d)):
            raise TypeError(f"unsupported target {name}: {type(modules[name])}")

    prepared_samples = []
    for index, sample_path in enumerate(sample_paths):
        sample = load_sample(sample_path)
        model_inputs, state = prepare_inputs(sample, cfg, model, processor, device)
        prepared_samples.append((model_inputs, state))
        print(f"[prepare] {index + 1}/{len(sample_paths)} {sample_path.name}")

    payload: dict[str, Any] = {
        "format_version": 1,
        "checkpoint": str(args.checkpoint),
        "bits": args.bits,
        "num_samples": len(sample_paths),
        "sample_names": [path.name for path in sample_paths],
        "targets": targets,
        "sequential": True,
        "profiles": {"awq": {"targets": {}}},
    }
    summary: dict[str, Any] = {
        "bits": args.bits,
        "num_samples": len(sample_paths),
        "sequential": True,
        "targets": {},
    }
    scale_handles = []

    try:
        for index, name in enumerate(targets):
            module = modules[name]
            captured = capture_target_inputs(
                module,
                prepared_samples,
                model,
                action_head,
                proprio_projector,
                cfg,
            )

            if isinstance(module, torch.nn.Linear):
                calibration_input = flatten_linear_inputs(
                    captured,
                    args.max_linear_rows,
                    args.seed + index,
                )
            else:
                calibration_input = select_conv_inputs(
                    captured,
                    args.max_conv_batches,
                )
            calibration_input = calibration_input.to(
                device=device,
                dtype=torch.float32,
            )

            entry = search_awq(
                module,
                calibration_input,
                args.bits,
                args.alpha_grid,
                args.clip_grid,
            )
            payload["profiles"]["awq"]["targets"][name] = entry
            summary["targets"][name] = {
                "index": index,
                "alpha": entry["alpha"],
                "clip_ratio": entry["clip_ratio"],
                "relative_mse": entry["relative_mse"],
            }

            scale_handles.append(apply_entry(module, entry))
            print(
                f"[sequential] {index + 1}/{len(targets)} {name} "
                f"alpha={entry['alpha']} clip={entry['clip_ratio']} "
                f"relative_mse={entry['relative_mse']:.8g}"
            )
            del captured, calibration_input
            torch.cuda.empty_cache()
    finally:
        for handle in scale_handles:
            handle.remove()

    entries = payload["profiles"]["awq"]["targets"]
    mean_mse = sum(entry["relative_mse"] for entry in entries.values()) / len(entries)
    payload["profiles"]["awq"]["mean_relative_mse"] = mean_mse
    summary["awq"] = {"mean_relative_mse": mean_mse}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    args.summary_json.write_text(json.dumps(summary, indent=2))

    print("summary:", args.summary_json)
    print("calibration:", args.output)
    print("SEQUENTIAL AWQ W2 CALIBRATION: PASS")


if __name__ == "__main__":
    main()
