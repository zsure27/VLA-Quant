from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qvla.action_jacobian_batch import initialize, load_sample, prepare_inputs
from qvla.calibrate_scaled_w2 import (
    input_channel_stat,
    make_input_scale,
    parse_csv_floats,
    profile_entry,
    quantize_rows_symmetric,
    transform_weight,
    weight_channel_stat,
)
from qvla.run_eval_scaled_w2 import make_input_scale_hook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select AWQ-style W2 scales by final normalized-action MSE."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
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
        default=(1.0, 0.9, 0.8),
    )
    return parser.parse_args()


@torch.no_grad()
def predict_normalized_actions(
    model: Any,
    action_head: Any,
    proprio_projector: Any,
    cfg: Any,
    prepared_samples: list[tuple[dict[str, torch.Tensor], np.ndarray]],
) -> list[torch.Tensor]:
    outputs = []
    for model_inputs, state in prepared_samples:
        _, hidden = model.predict_action(
            **model_inputs,
            unnorm_key=cfg.unnorm_key,
            do_sample=False,
            proprio=state,
            proprio_projector=proprio_projector,
            action_head=action_head,
            noisy_action_projector=None,
            use_film=False,
        )
        outputs.append(action_head.predict_action(hidden).detach().float().cpu())
    return outputs


@torch.no_grad()
def capture_inputs(
    model: Any,
    action_head: Any,
    proprio_projector: Any,
    cfg: Any,
    prepared_samples: list[tuple[dict[str, torch.Tensor], np.ndarray]],
    target: torch.nn.Module,
) -> list[torch.Tensor]:
    captured: list[torch.Tensor] = []

    def hook(_module, args):
        captured.append(args[0].detach().float().cpu())

    handle = target.register_forward_pre_hook(hook)
    try:
        predict_normalized_actions(
            model,
            action_head,
            proprio_projector,
            cfg,
            prepared_samples,
        )
    finally:
        handle.remove()
    if (
        len(captured) < len(prepared_samples)
        or len(captured) % len(prepared_samples) != 0
    ):
        raise RuntimeError(
            f"captured {len(captured)} inputs for {len(prepared_samples)} samples"
        )
    print(
        f"[capture] calls_per_sample={len(captured) // len(prepared_samples)} "
        f"total_inputs={len(captured)}"
    )
    return captured


def activation_max(module: torch.nn.Module, inputs: list[torch.Tensor]) -> torch.Tensor:
    if isinstance(module, torch.nn.Linear):
        result = torch.stack(
            [value.reshape(-1, value.shape[-1]).abs().amax(dim=0) for value in inputs]
        ).amax(dim=0)
    elif isinstance(module, torch.nn.Conv2d):
        result = torch.stack(
            [value.abs().amax(dim=(0, 2, 3)) for value in inputs]
        ).amax(dim=0)
    else:
        raise TypeError(type(module))
    return result.clamp_min(1e-6)


def action_relative_mse(
    candidate: list[torch.Tensor],
    reference: list[torch.Tensor],
) -> float:
    numerator = sum(
        (left - right).square().sum().item()
        for left, right in zip(candidate, reference)
    )
    denominator = sum(right.square().sum().item() for right in reference)
    return numerator / max(denominator, 1e-12)


@torch.no_grad()
def apply_candidate(
    module: torch.nn.Module,
    original_weight: torch.Tensor,
    input_scale: torch.Tensor,
    clip_ratio: float,
    bits: int,
) -> tuple[torch.Tensor, Any]:
    scaled_weight = transform_weight(module, original_weight.float(), input_scale)
    quantized_weight = quantize_rows_symmetric(scaled_weight, bits, clip_ratio)
    module.weight.copy_(quantized_weight.to(module.weight.dtype))
    handle = module.register_forward_pre_hook(
        make_input_scale_hook(module, input_scale.float())
    )
    return quantized_weight, handle


def main() -> None:
    args = parse_args()
    if args.bits != 2:
        raise ValueError("This controlled experiment is fixed to W2A16")
    targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    if not targets or len(targets) != len(set(targets)):
        raise ValueError("targets must be non-empty and unique")

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
        prepared_samples.append(
            prepare_inputs(sample, cfg, model, processor, device)
        )
        print(f"[prepare] {index + 1}/{len(sample_paths)} {sample_path.name}")

    reference_actions = predict_normalized_actions(
        model,
        action_head,
        proprio_projector,
        cfg,
        prepared_samples,
    )
    payload: dict[str, Any] = {
        "format_version": 1,
        "checkpoint": str(args.checkpoint),
        "bits": args.bits,
        "num_samples": len(sample_paths),
        "sample_names": [path.name for path in sample_paths],
        "targets": targets,
        "action_aware": True,
        "profiles": {"awq": {"targets": {}}},
    }
    summary: dict[str, Any] = {
        "bits": args.bits,
        "num_samples": len(sample_paths),
        "action_aware": True,
        "targets": {},
    }
    accepted_handles = []

    try:
        for index, name in enumerate(targets):
            module = modules[name]
            inputs = capture_inputs(
                model,
                action_head,
                proprio_projector,
                cfg,
                prepared_samples,
                module,
            )
            activation_stat = activation_max(module, inputs).to(device)
            original_weight = module.weight.detach().clone()
            weight_stat = weight_channel_stat(
                module,
                original_weight.float(),
                "awq",
            )

            best: tuple[float, float, float, torch.Tensor, torch.Tensor] | None = None
            for alpha in args.alpha_grid:
                input_scale = make_input_scale(
                    activation_stat,
                    weight_stat,
                    alpha,
                )
                for clip_ratio in args.clip_grid:
                    quantized_weight, handle = apply_candidate(
                        module,
                        original_weight,
                        input_scale,
                        clip_ratio,
                        args.bits,
                    )
                    candidate_actions = predict_normalized_actions(
                        model,
                        action_head,
                        proprio_projector,
                        cfg,
                        prepared_samples,
                    )
                    score = action_relative_mse(candidate_actions, reference_actions)
                    handle.remove()
                    module.weight.copy_(original_weight)

                    if best is None or score < best[0]:
                        best = (
                            score,
                            alpha,
                            clip_ratio,
                            input_scale.detach().clone(),
                            quantized_weight.detach().clone(),
                        )

            assert best is not None
            entry = profile_entry(
                "awq",
                args.bits,
                best[1],
                best[2],
                best[0],
                best[3],
                best[4],
                module,
            )
            entry["selection_metric"] = "normalized_action_relative_mse"
            payload["profiles"]["awq"]["targets"][name] = entry
            summary["targets"][name] = {
                "index": index,
                "alpha": entry["alpha"],
                "clip_ratio": entry["clip_ratio"],
                "action_relative_mse": entry["relative_mse"],
            }

            _, accepted = apply_candidate(
                module,
                original_weight,
                entry["input_scale"].to(device),
                entry["clip_ratio"],
                args.bits,
            )
            accepted_handles.append(accepted)
            print(
                f"[action-aware] {index + 1}/{len(targets)} {name} "
                f"alpha={entry['alpha']} clip={entry['clip_ratio']} "
                f"action_relative_mse={entry['relative_mse']:.8g}"
            )
            del inputs, activation_stat, original_weight, weight_stat
            torch.cuda.empty_cache()
    finally:
        for handle in accepted_handles:
            handle.remove()

    entries = payload["profiles"]["awq"]["targets"]
    mean_score = sum(entry["relative_mse"] for entry in entries.values()) / len(entries)
    payload["profiles"]["awq"]["mean_relative_mse"] = mean_score
    summary["awq"] = {"mean_action_relative_mse": mean_score}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    args.summary_json.write_text(json.dumps(summary, indent=2))
    print("summary:", args.summary_json)
    print("calibration:", args.output)
    print("ACTION-AWARE AWQ W2 CALIBRATION: PASS")


if __name__ == "__main__":
    main()
