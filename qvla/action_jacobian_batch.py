from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model
from experiments.robot.openvla_utils import normalize_proprio, prepare_images_for_vla


BITS = (2, 4, 8)


def required_score_keys() -> set[str]:
    return {"prune_0"} | {
        f"{mode}_{bit}"
        for mode in ("symmetric", "asymmetric")
        for bit in BITS
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute action-space quantization sensitivity for many layers per model load."
    )
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--probes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--gradient-mode",
        choices=("independent", "shared"),
        default="independent",
        help="independent matches the single-layer reference; shared is faster but may change early-layer values",
    )
    return parser.parse_args()


def quantize_rows_symmetric(weight: torch.Tensor, bits: int) -> torch.Tensor:
    original_shape = weight.shape
    weight = weight.reshape(weight.shape[0], -1)
    qmax = (1 << (bits - 1)) - 1
    scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    quantized = torch.round(weight / scale).clamp(-(qmax + 1), qmax)
    return (quantized * scale).reshape(original_shape)


def quantize_rows_asymmetric(weight: torch.Tensor, bits: int) -> torch.Tensor:
    original_shape = weight.shape
    weight = weight.reshape(weight.shape[0], -1)
    qmin, qmax = 0, (1 << bits) - 1
    minimum = weight.amin(dim=1, keepdim=True)
    maximum = weight.amax(dim=1, keepdim=True)
    scale = ((maximum - minimum) / (qmax - qmin)).clamp_min(1e-8)
    zero_point = torch.round(qmin - minimum / scale).clamp(qmin, qmax)
    quantized = torch.round(weight / scale + zero_point).clamp(qmin, qmax)
    return ((quantized - zero_point) * scale).reshape(original_shape)


def output_channels(module: torch.nn.Module) -> int:
    if isinstance(module, torch.nn.Linear):
        return module.out_features
    if isinstance(module, torch.nn.Conv2d):
        return module.out_channels
    raise TypeError(f"Unsupported target module: {type(module)}")


def quantization_delta(
    module: torch.nn.Module,
    layer_input: torch.Tensor,
    weight_error: torch.Tensor,
) -> torch.Tensor:
    if isinstance(module, torch.nn.Linear):
        return F.linear(layer_input, weight_error)
    if isinstance(module, torch.nn.Conv2d):
        return F.conv2d(
            layer_input,
            weight_error,
            bias=None,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )
    raise TypeError(f"Unsupported target module: {type(module)}")


def channel_reduce_dims(module: torch.nn.Module, tensor: torch.Tensor) -> tuple[int, ...]:
    channel_dim = tensor.ndim - 1 if isinstance(module, torch.nn.Linear) else 1
    return tuple(dim for dim in range(tensor.ndim) if dim != channel_dim)


def differentiable_unnormalize(model: Any, key: str, actions: torch.Tensor) -> torch.Tensor:
    stats = model.get_action_stats(key)
    if "q99" in stats:
        high = np.asarray(stats["q99"], dtype=np.float32)
        low = np.asarray(stats["q01"], dtype=np.float32)
    else:
        high = np.asarray(stats["max"], dtype=np.float32)
        low = np.asarray(stats["min"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(high, dtype=bool)))
    high_t = torch.as_tensor(high, device=actions.device)
    low_t = torch.as_tensor(low, device=actions.device)
    mask_t = torch.as_tensor(mask, device=actions.device, dtype=torch.bool)
    unnormalized = 0.5 * (actions + 1.0) * (high_t - low_t + 1e-8) + low_t
    return torch.where(mask_t, unnormalized, actions)


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace(".", "__")


def valid_output(path: Path, target: str) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        payload = torch.load(path, map_location="cpu")
        scores = payload.get("scores", {})
        return (
            payload.get("target") == target
            and payload.get("active") is True
            and required_score_keys().issubset(scores)
            and all(torch.isfinite(value).all() for value in scores.values())
        )
    except Exception:
        return False


def read_targets(path: Path) -> list[str]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            targets = payload
        elif isinstance(payload, dict) and "connected_targets" in payload:
            connected = payload["connected_targets"]
            targets = connected if isinstance(connected, list) else connected.get("names", [])
        else:
            targets = payload.get("targets", [])
    else:
        targets = [line.strip() for line in path.read_text().splitlines()]
    targets = [str(name) for name in targets if str(name).strip() and not str(name).startswith("#")]
    if not targets:
        raise ValueError(f"No targets found in {path}")
    if len(targets) != len(set(targets)):
        raise ValueError("Target list contains duplicates")
    return targets


def load_sample(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as sample:
        return {
            "image": sample["image"],
            "wrist_image": sample["wrist_image"],
            "state": sample["state"].astype(np.float32),
            "dataset_action": sample["action"].astype(np.float32),
            "instruction": str(sample["instruction"].item()),
        }


def initialize(checkpoint: str, seed: int):
    cfg = GenerateConfig(
        pretrained_checkpoint=checkpoint,
        task_suite_name="libero_spatial",
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=True,
        seed=seed,
    )
    model, action_head, proprio_projector, _, processor = initialize_model(cfg)
    for module in (model, action_head, proprio_projector):
        module.requires_grad_(False)
        module.eval()
    model.vision_backbone.set_num_images_in_input(2)
    return cfg, model, action_head, proprio_projector, processor


def prepare_inputs(sample: dict[str, Any], cfg: Any, model: Any, processor: Any, device: torch.device):
    images = prepare_images_for_vla([sample["image"], sample["wrist_image"]], cfg)
    prompt = f"In: What action should the robot take to {sample['instruction'].lower()}?\nOut:"
    primary = processor(prompt, images[0], return_tensors="pt").to(device, dtype=torch.bfloat16)
    wrist = processor(prompt, images[1], return_tensors="pt").to(device, dtype=torch.bfloat16)
    pixels = torch.cat([primary["pixel_values"], wrist["pixel_values"]], dim=1).contiguous()
    expected_channels = 6 * cfg.num_images_in_input
    if pixels.shape[1] != expected_channels:
        raise RuntimeError(f"Expected {expected_channels} pixel channels, got {pixels.shape[1]}")
    stats = model.norm_stats[cfg.unnorm_key]["proprio"]
    state = normalize_proprio(sample["state"].copy(), stats)
    inputs = {
        "input_ids": primary["input_ids"],
        "attention_mask": primary["attention_mask"],
        "pixel_values": pixels,
    }
    return inputs, state


def run_group(
    names: list[str], modules: dict[str, torch.nn.Module], model: Any, action_head: Any,
    proprio_projector: Any, cfg: Any, model_inputs: dict[str, torch.Tensor], state: np.ndarray,
    sample: dict[str, Any], output_dir: Path, probes: int, seed: int, device: torch.device,
    gradient_mode: str,
) -> None:
    captured: dict[str, dict[str, torch.Tensor]] = {name: {} for name in names}
    handles = []

    def make_hook(name: str):
        def hook(_module, inputs, output):
            captured[name]["input"] = inputs[0].detach()
            if not output.requires_grad:
                output.requires_grad_(True)
            output.retain_grad()
            captured[name]["output"] = output
            return output
        return hook

    for name in names:
        handles.append(modules[name].register_forward_hook(make_hook(name)))
    try:
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
    finally:
        for handle in handles:
            handle.remove()

    normalized_actions = action_head.predict_action(hidden)
    physical_actions = differentiable_unnormalize(model, cfg.unnorm_key, normalized_actions)
    active = [name for name in names if "output" in captured[name]]
    if len(active) != len(names):
        missing = sorted(set(names) - set(active))
        raise RuntimeError(f"Targets were not called: {missing}")

    deltas: dict[str, dict[str, torch.Tensor]] = {}
    scores: dict[str, dict[str, torch.Tensor]] = {}
    quantizers = (("symmetric", quantize_rows_symmetric), ("asymmetric", quantize_rows_asymmetric))
    for name in active:
        target = modules[name]
        layer_input = captured[name]["input"].float()
        weight = target.weight.detach().float()
        deltas[name] = {}
        scores[name] = {}
        deltas[name]["prune_0"] = quantization_delta(target, layer_input, -weight)
        scores[name]["prune_0"] = torch.zeros(
            output_channels(target), device=device, dtype=torch.float32
        )
        for mode, quantizer in quantizers:
            for bits in BITS:
                key = f"{mode}_{bits}"
                quantized = quantizer(weight, bits)
                deltas[name][key] = quantization_delta(target, layer_input, quantized - weight)
                scores[name][key] = torch.zeros(
                    output_channels(target), device=device, dtype=torch.float32
                )
                del quantized
        del layer_input, weight

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    for probe_index in range(probes):
        random_vector = torch.randint(
            0, 2, physical_actions.shape, generator=generator, device=device, dtype=torch.int64
        ).float() * 2.0 - 1.0
        objective = (physical_actions.float() * random_vector).sum()
        if gradient_mode == "shared":
            for name in active:
                captured[name]["output"].grad = None
            objective.backward(retain_graph=probe_index < probes - 1)
            gradients = [captured[name]["output"].grad for name in active]
        else:
            gradients = []
            for target_index, name in enumerate(active):
                final_gradient = probe_index == probes - 1 and target_index == len(active) - 1
                gradient = torch.autograd.grad(
                    objective,
                    captured[name]["output"],
                    retain_graph=not final_gradient,
                    allow_unused=True,
                )[0]
                gradients.append(gradient)

        for name, gradient in zip(active, gradients):
            if gradient is None:
                raise RuntimeError(f"Target is disconnected from actions: {name}")
            gradient = gradient.float()
            reduce_dims = channel_reduce_dims(modules[name], gradient)
            for key, delta in deltas[name].items():
                projected = (gradient * delta).sum(dim=reduce_dims)
                scores[name][key] += projected.square() / probes

    for name in active:
        cpu_scores = {key: value.detach().cpu() for key, value in sorted(scores[name].items())}
        for value in cpu_scores.values():
            if not torch.isfinite(value).all() or not (value >= 0).all():
                raise RuntimeError(f"Invalid score values for {name}")
        result = {
            "target": name,
            "instruction": sample["instruction"],
            "active": True,
            "probes": probes,
            "scores": cpu_scores,
            "predicted_actions": physical_actions.detach().cpu(),
            "dataset_action": torch.from_numpy(sample["dataset_action"]),
        }
        destination = output_dir / f"{safe_name(name)}.pt"
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(result, temporary)
        temporary.replace(destination)


def main() -> None:
    args = parse_args()
    if args.group_size < 1 or args.probes < 1:
        raise ValueError("group-size and probes must be positive")
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample = load_sample(args.sample)
    targets = read_targets(args.targets)
    cfg, model, action_head, proprio_projector, processor = initialize(args.checkpoint, args.seed)
    modules = dict(model.named_modules())
    for name in targets:
        if name not in modules:
            raise KeyError(f"Target not found: {name}")
        if not isinstance(modules[name], (torch.nn.Linear, torch.nn.Conv2d)):
            raise TypeError(f"Expected Linear/Conv2d for {name}, got {type(modules[name])}")
    model_inputs, state = prepare_inputs(sample, cfg, model, processor, device)
    pending = [
        name for name in targets
        if not valid_output(args.output_dir / f"{safe_name(name)}.pt", name)
    ]
    print(
        f"targets={len(targets)} pending={len(pending)} "
        f"group_size={args.group_size} gradient_mode={args.gradient_mode}"
    )

    index = 0
    group_size = min(args.group_size, max(1, len(pending)))
    while index < len(pending):
        group = pending[index:index + group_size]
        print(f"[batch] {index + 1}-{index + len(group)}/{len(pending)} size={len(group)}")
        try:
            run_group(
                group, modules, model, action_head, proprio_projector, cfg, model_inputs, state,
                sample, args.output_dir, args.probes, args.seed, device, args.gradient_mode,
            )
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            if len(group) == 1:
                raise
            group_size = max(1, len(group) // 2)
            print(f"[batch] CUDA OOM; retrying with group_size={group_size}")
            continue
        index += len(group)
        gc.collect()
        torch.cuda.empty_cache()

    completed = sum(
        valid_output(args.output_dir / f"{safe_name(name)}.pt", name)
        for name in targets
    )
    print(f"completed={completed}/{len(targets)}")
    if completed != len(targets):
        raise RuntimeError("Batch finished with missing outputs")
    print("ACTION-JACOBIAN BATCH: PASS")


if __name__ == "__main__":
    main()
