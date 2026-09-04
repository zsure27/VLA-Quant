import argparse
import json
import os
import sys

import torch
import torch.nn as nn


@torch.no_grad()
def fake_quantize_symmetric(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 16:
        return x
    if bits <= 0:
        return torch.zeros_like(x)

    qmax = (1 << (bits - 1)) - 1
    scale = x.abs().max().clamp_min(1e-8) / qmax
    q = torch.clamp(
        torch.round(x / scale),
        min=-(qmax + 1),
        max=qmax,
    )
    return q * scale


@torch.no_grad()
def quantize_component(root: nn.Module, bits: int, label: str):
    modules = 0
    parameters = 0

    for name, module in root.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue

        weight = module.weight.data

        # 16 位控制组不得写入检查点权重。
        if bits < 16:
            for channel in range(weight.shape[0]):
                weight[channel] = fake_quantize_symmetric(
                    weight[channel], bits
                )

        modules += 1
        parameters += weight.numel()

    print(
        f"[component-w] scope={label} bits={bits} "
        f"modules={modules} parameters={parameters}"
    )
    return modules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument(
        "--component_scopes",
        required=True,
        help="projector, proprio, action, or comma-separated combination",
    )
    parser.add_argument("--component_bits", type=int, required=True)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--num_trials_per_task", type=int, default=1)
    parser.add_argument("--local_log_dir", required=True)
    parser.add_argument(
        "--libero_root",
        default=os.environ.get("LIBERO_ROOT", ""),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--attention",
        choices=["default", "eager", "flash_attention_2"],
        default="default",
    )
    args = parser.parse_args()

    if args.component_bits not in (2, 4, 8, 16):
        raise ValueError("component_bits must be 2, 4, 8, or 16")

    scopes = {
        item.strip()
        for item in args.component_scopes.split(",")
        if item.strip()
    }
    allowed = {"projector", "proprio", "action"}
    unknown = scopes - allowed
    if unknown:
        raise ValueError(f"unknown component scopes: {sorted(unknown)}")

    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    sys.path.insert(0, repo_root)

    if args.libero_root:
        sys.path.insert(0, args.libero_root)

    import experiments.robot.libero.run_libero_eval as L
    import experiments.robot.robot_utils as R
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForVision2Seq,
        AutoProcessor,
    )
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import (
        OpenVLAForActionPrediction,
    )
    from prismatic.extern.hf.processing_prismatic import (
        PrismaticImageProcessor,
        PrismaticProcessor,
    )

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(
        OpenVLAConfig, PrismaticImageProcessor
    )
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(
        OpenVLAConfig, OpenVLAForActionPrediction
    )

    device = torch.device("cuda")
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if args.attention != "default":
        load_kwargs["attn_implementation"] = args.attention

    print(
        "[component-w] num_images_in_input=2, "
        f"attention={args.attention}"
    )

    model = AutoModelForVision2Seq.from_pretrained(
        args.pretrained_checkpoint,
        **load_kwargs,
    ).to(device)
    model.eval()

    stats_path = os.path.join(
        args.pretrained_checkpoint,
        "dataset_statistics.json",
    )
    with open(stats_path) as handle:
        model.norm_stats = json.load(handle)

    if "projector" in scopes:
        quantize_component(
            model.projector,
            args.component_bits,
            "projector",
        )

    original_get_model = R.get_model
    original_l_get_model = getattr(L, "get_model", None)
    original_get_action_head = L.get_action_head
    original_get_proprio_projector = L.get_proprio_projector

    def patched_get_model(cfg):
        print("[component-w] using preloaded VLA model")
        return model

    def patched_get_action_head(*call_args, **call_kwargs):
        component = original_get_action_head(
            *call_args, **call_kwargs
        )
        if "action" in scopes:
            quantize_component(
                component,
                args.component_bits,
                "action",
            )
        return component

    def patched_get_proprio_projector(*call_args, **call_kwargs):
        component = original_get_proprio_projector(
            *call_args, **call_kwargs
        )
        if "proprio" in scopes:
            quantize_component(
                component,
                args.component_bits,
                "proprio",
            )
        return component

    R.get_model = patched_get_model
    if original_l_get_model is not None:
        L.get_model = patched_get_model
    L.get_action_head = patched_get_action_head
    L.get_proprio_projector = patched_get_proprio_projector

    argv_backup = list(sys.argv)

    try:
        sys.argv = [
            sys.argv[0],
            "--model_family", "openvla",
            "--pretrained_checkpoint", args.pretrained_checkpoint,
            "--num_images_in_input", "2",
            "--task_suite_name", args.task_suite_name,
            "--num_trials_per_task",
            str(args.num_trials_per_task),
            "--local_log_dir", args.local_log_dir,
            "--center_crop", "True",
            "--seed", str(args.seed),
        ]
        os.makedirs(args.local_log_dir, exist_ok=True)
        L.eval_libero()
    finally:
        sys.argv = argv_backup
        R.get_model = original_get_model
        if original_l_get_model is not None:
            L.get_model = original_l_get_model
        L.get_action_head = original_get_action_head
        L.get_proprio_projector = original_get_proprio_projector


if __name__ == "__main__":
    main()
