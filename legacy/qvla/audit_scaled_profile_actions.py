from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from qvla.action_jacobian_batch import (
    differentiable_unnormalize,
    initialize,
    load_sample,
    prepare_inputs,
)
from qvla.run_eval_scaled_w2 import apply_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit BF16 versus a scaled fake-W2 profile in action space."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples-dir", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--method", default="awq")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


@torch.no_grad()
def predict(
    model: Any,
    action_head: Any,
    proprio_projector: Any,
    cfg: Any,
    prepared_samples: list[tuple[dict[str, torch.Tensor], np.ndarray]],
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = []
    physical = []
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
        actions = action_head.predict_action(hidden).detach().float()
        normalized.append(actions.cpu())
        physical.append(
            differentiable_unnormalize(model, cfg.unnorm_key, actions).cpu()
        )
    return torch.cat(normalized, dim=0), torch.cat(physical, dim=0)


def metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    difference = candidate - reference
    denominator = reference.square().mean().clamp_min(1e-12)
    return {
        "relative_mse": float((difference.square().mean() / denominator).item()),
        "mae": float(difference.abs().mean().item()),
        "max_abs": float(difference.abs().max().item()),
        "cosine": float(
            F.cosine_similarity(
                candidate.reshape(1, -1),
                reference.reshape(1, -1),
                dim=1,
                eps=1e-12,
            ).item()
        ),
    }


def main() -> None:
    args = parse_args()
    paths = sorted(args.samples_dir.glob("sample-*.npz"))
    paths = paths[args.start_index : args.start_index + args.num_samples]
    if len(paths) != args.num_samples:
        raise RuntimeError(
            f"requested {args.num_samples} samples from index {args.start_index}, "
            f"found {len(paths)}"
        )

    device = torch.device("cuda:0")
    cfg, model, action_head, proprio_projector, processor = initialize(
        args.checkpoint,
        args.seed,
    )
    prepared = []
    for path in paths:
        sample = load_sample(path)
        prepared.append(prepare_inputs(sample, cfg, model, processor, device))

    reference_normalized, reference_physical = predict(
        model,
        action_head,
        proprio_projector,
        cfg,
        prepared,
    )
    handles = apply_profile(model, args.calibration, args.method)
    try:
        candidate_normalized, candidate_physical = predict(
            model,
            action_head,
            proprio_projector,
            cfg,
            prepared,
        )
    finally:
        for handle in handles:
            handle.remove()

    per_sample = []
    for index, path in enumerate(paths):
        per_sample.append(
            {
                "sample": path.name,
                "normalized": metrics(
                    candidate_normalized[index], reference_normalized[index]
                ),
                "physical": metrics(
                    candidate_physical[index], reference_physical[index]
                ),
            }
        )

    result = {
        "calibration": str(args.calibration),
        "start_index": args.start_index,
        "num_samples": args.num_samples,
        "sample_names": [path.name for path in paths],
        "normalized": metrics(candidate_normalized, reference_normalized),
        "physical": metrics(candidate_physical, reference_physical),
        "per_sample": per_sample,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2))

    print("samples:", args.start_index, "to", args.start_index + args.num_samples - 1)
    print("normalized:", result["normalized"])
    print("physical:", result["physical"])
    print("saved:", args.output_json)
    print("SCALED PROFILE ACTION AUDIT: PASS")


if __name__ == "__main__":
    main()
