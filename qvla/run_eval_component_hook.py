import argparse
import os
import sys

import torch
import torch.nn as nn


@torch.no_grad()
def fake_quantize_symmetric(x: torch.Tensor, bits: int):
    if bits >= 16:
        return x
    if bits <= 0:
        return torch.zeros_like(x)

    qmax = (1 << (bits - 1)) - 1
    scale = x.abs().max().clamp_min(1e-8) / qmax
    quantized = torch.clamp(
        torch.round(x / scale),
        min=-(qmax + 1),
        max=qmax,
    )
    return quantized * scale


@torch.no_grad()
def quantize_component(root: nn.Module, bits: int, label: str):
    selector_text = os.environ.get(
        "QVLA_COMPONENT_MODULE", ""
    ).strip()
    selectors = {
        item.strip()
        for item in selector_text.split(",")
        if item.strip()
    }

    module_count = 0
    parameter_count = 0
    selected_names = []

    for name, module in root.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue

        qualified_name = (
            f"{label}.{name}" if name else label
        )

        if selectors and qualified_name not in selectors:
            continue

        weight = module.weight.data

        # W16 恒等控制组不写入权重。
        if bits < 16:
            for channel in range(weight.shape[0]):
                weight[channel] = fake_quantize_symmetric(
                    weight[channel], bits
                )

        module_count += 1
        parameter_count += weight.numel()
        selected_names.append(qualified_name)

    if selectors and module_count == 0:
        raise RuntimeError(
            "no selected component module found for "
            f"scope={label}; requested={sorted(selectors)}"
        )

    print(
        f"[component-hook] scope={label} bits={bits} "
        f"modules={module_count} parameters={parameter_count}"
    )
    print(
        "[component-hook] selected="
        + ",".join(selected_names)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--gates_path", required=True)
    parser.add_argument("--component_scopes", required=True)
    parser.add_argument("--component_bits", type=int, required=True)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--num_trials_per_task", type=int, default=1)
    parser.add_argument("--local_log_dir", required=True)
    parser.add_argument(
        "--libero_root",
        default=os.environ.get("LIBERO_ROOT", ""),
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.component_bits not in (2, 4, 8, 16):
        raise ValueError("component_bits must be 2, 4, 8, or 16")

    scopes = {
        item.strip()
        for item in args.component_scopes.split(",")
        if item.strip() and item.strip() != "none"
    }
    allowed = {"projector", "proprio", "action"}
    unknown = scopes - allowed
    if unknown:
        raise ValueError(f"unknown component scopes: {sorted(unknown)}")

    import experiments.robot.libero.run_libero_eval as libero_eval
    import qvla.inject_fake_w as fake_w
    import qvla.run_eval as base_runner

    original_inject = fake_w.inject_qvla_weight_fake_quant
    original_action = libero_eval.get_action_head
    original_proprio = libero_eval.get_proprio_projector

    def patched_inject(model, *call_args, **call_kwargs):
        count = original_inject(
            model, *call_args, **call_kwargs
        )

        if "projector" in scopes:
            if not hasattr(model, "projector"):
                raise AttributeError("VLA model has no projector")
            quantize_component(
                model.projector,
                args.component_bits,
                "projector",
            )

        return count

    def patched_action(*call_args, **call_kwargs):
        component = original_action(*call_args, **call_kwargs)

        if "action" in scopes:
            quantize_component(
                component,
                args.component_bits,
                "action",
            )

        return component

    def patched_proprio(*call_args, **call_kwargs):
        component = original_proprio(*call_args, **call_kwargs)

        if "proprio" in scopes:
            quantize_component(
                component,
                args.component_bits,
                "proprio",
            )

        return component

    fake_w.inject_qvla_weight_fake_quant = patched_inject
    libero_eval.get_action_head = patched_action
    libero_eval.get_proprio_projector = patched_proprio

    argv_backup = list(sys.argv)

    try:
        sys.argv = [
            sys.argv[0],
            "--pretrained_checkpoint",
            args.pretrained_checkpoint,
            "--gates_path",
            args.gates_path,
            "--task_suite_name",
            args.task_suite_name,
            "--num_trials_per_task",
            str(args.num_trials_per_task),
            "--local_log_dir",
            args.local_log_dir,
            "--libero_root",
            args.libero_root,
            "--seed",
            str(args.seed),
        ]

        print(
            "[component-hook] delegating model loading "
            "to verified qvla/run_eval.py"
        )
        base_runner.main()
    finally:
        sys.argv = argv_backup
        fake_w.inject_qvla_weight_fake_quant = original_inject
        libero_eval.get_action_head = original_action
        libero_eval.get_proprio_projector = original_proprio


if __name__ == "__main__":
    main()
