from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate identity, RTN, SmoothQuant-style, or AWQ-style W2A16."
    )
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--calibration", type=Path, action="append")
    parser.add_argument(
        "--method",
        required=True,
        choices=("identity", "rtn", "smoothquant", "awq"),
    )
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--num_trials_per_task", type=int, default=1)
    parser.add_argument("--local_log_dir", required=True)
    parser.add_argument("--libero_root", default=os.environ.get("LIBERO_ROOT", ""))
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def make_input_scale_hook(
    module: torch.nn.Module,
    input_scale: torch.Tensor,
):
    def hook(_module, args):
        if not args:
            raise RuntimeError("scaled module received no positional input")
        value = args[0]
        scale = input_scale.to(device=value.device, dtype=value.dtype)
        if isinstance(module, torch.nn.Linear):
            shape = [1] * (value.ndim - 1) + [scale.numel()]
        elif isinstance(module, torch.nn.Conv2d):
            shape = [1, scale.numel()] + [1] * (value.ndim - 2)
        else:
            raise TypeError(type(module))
        scaled = value / scale.view(shape)
        return (scaled,) + tuple(args[1:])
    return hook


@torch.no_grad()
def apply_profile(
    model: torch.nn.Module,
    calibration_path: Path,
    method: str,
) -> list[Any]:
    payload = torch.load(calibration_path, map_location="cpu")
    if payload.get("format_version") != 1:
        raise RuntimeError("unsupported calibration format")
    if method not in payload.get("profiles", {}):
        raise KeyError(f"profile not found: {method}")

    profile = payload["profiles"][method]
    entries = profile.get("targets", {})
    if not entries:
        raise RuntimeError(f"empty calibration profile: {method}")

    modules = dict(model.named_modules())
    handles = []

    for name, entry in entries.items():
        if name not in modules:
            raise KeyError(f"target not found in model: {name}")
        module = modules[name]
        if not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            raise TypeError(f"unsupported target type for {name}: {type(module)}")

        quantized_weight = entry["quantized_weight"]
        input_scale = entry["input_scale"].float()
        if tuple(quantized_weight.shape) != tuple(module.weight.shape):
            raise RuntimeError(
                f"weight shape mismatch for {name}: "
                f"{tuple(quantized_weight.shape)} != {tuple(module.weight.shape)}"
            )

        expected_inputs = module.weight.shape[1]
        if input_scale.numel() != expected_inputs:
            raise RuntimeError(
                f"input scale mismatch for {name}: "
                f"{input_scale.numel()} != {expected_inputs}"
            )

        module.weight.copy_(
            quantized_weight.to(
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
        )
        handles.append(
            module.register_forward_pre_hook(
                make_input_scale_hook(module, input_scale)
            )
        )

        print(
            f"[scaled-w2] method={method} target={name} "
            f"alpha={entry['alpha']} clip={entry['clip_ratio']} "
            f"relative_mse={entry['relative_mse']:.8g}"
        )

    print(
        f"[scaled-w2] applied method={method} "
        f"targets={len(entries)} mean_relative_mse="
        f"{profile.get('mean_relative_mse', float('nan')):.8g}"
    )
    return handles


def main() -> None:
    args = parse_args()
    if args.method != "identity" and args.calibration is None:
        raise ValueError("--calibration is required unless --method identity")

    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if args.libero_root and os.path.isdir(args.libero_root):
        if args.libero_root not in sys.path:
            sys.path.insert(0, args.libero_root)

    import experiments.robot.libero.run_libero_eval as L
    import experiments.robot.robot_utils as R
    from experiments.robot.libero.run_libero_eval import eval_libero
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import (
        PrismaticImageProcessor,
        PrismaticProcessor,
    )
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForVision2Seq,
        AutoProcessor,
    )

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForVision2Seq.from_pretrained(
        args.pretrained_checkpoint,
        torch_dtype=(torch.bfloat16 if device.type == "cuda" else torch.float32),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    model.vision_backbone.set_num_images_in_input(2)
    print("[scaled-w2] num_images_in_input=2, attention=checkpoint default")

    statistics_path = os.path.join(
        args.pretrained_checkpoint, "dataset_statistics.json"
    )
    if os.path.isfile(statistics_path):
        with open(statistics_path, "r") as file:
            model.norm_stats = json.load(file)

    handles = []
    if args.method == "identity":
        print("[scaled-w2] identity control: no weights or forwards modified")
    else:
        for calibration_path in args.calibration:
            handles.extend(
                apply_profile(model, calibration_path, args.method)
            )

    original_get_model = R.get_model
    original_libero_get_model = getattr(L, "get_model", None)

    def patched_get_model(_cfg):
        print("[scaled-w2] using preloaded model")
        return model

    R.get_model = patched_get_model
    if original_libero_get_model is not None:
        L.get_model = patched_get_model

    argv_backup = list(sys.argv)
    try:
        sys.argv = [
            sys.argv[0],
            "--model_family",
            "openvla",
            "--pretrained_checkpoint",
            str(args.pretrained_checkpoint),
            "--task_suite_name",
            str(args.task_suite_name),
            "--num_trials_per_task",
            str(args.num_trials_per_task),
            "--local_log_dir",
            str(args.local_log_dir),
            "--center_crop",
            str(True),
            "--seed",
            str(args.seed),
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
