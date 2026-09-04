from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model
from experiments.robot.openvla_utils import normalize_proprio, prepare_images_for_vla


def quantize_rows_symmetric(weight: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = (1 << (bits - 1)) - 1
    scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    quantized = torch.round(weight / scale).clamp(-(qmax + 1), qmax)
    return quantized * scale


def quantize_rows_asymmetric(weight: torch.Tensor, bits: int) -> torch.Tensor:
    qmin, qmax = 0, (1 << bits) - 1
    minimum = weight.amin(dim=1, keepdim=True)
    maximum = weight.amax(dim=1, keepdim=True)
    scale = ((maximum - minimum) / (qmax - qmin)).clamp_min(1e-8)
    zero_point = torch.round(qmin - minimum / scale).clamp(qmin, qmax)
    quantized = torch.round(weight / scale + zero_point).clamp(qmin, qmax)
    return (quantized - zero_point) * scale


def differentiable_unnormalize(model, key: str, actions: torch.Tensor) -> torch.Tensor:
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


def main() -> None:
    sample_path, checkpoint, output_path, target_name = sys.argv[1:5]
    device = torch.device("cuda:0")

    torch.manual_seed(7)
    np.random.seed(7)

    with np.load(sample_path, allow_pickle=False) as sample:
        image = sample["image"]
        wrist_image = sample["wrist_image"]
        state = sample["state"].astype(np.float32)
        dataset_action = sample["action"].astype(np.float32)
        instruction = str(sample["instruction"].item())

    cfg = GenerateConfig(
        pretrained_checkpoint=checkpoint,
        task_suite_name="libero_spatial",
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=True,
        seed=7,
    )

    model, action_head, proprio_projector, _, processor = initialize_model(cfg)

    model.requires_grad_(False)
    action_head.requires_grad_(False)
    proprio_projector.requires_grad_(False)

    model.eval()
    action_head.eval()
    proprio_projector.eval()
    model.vision_backbone.set_num_images_in_input(2)

    modules = dict(model.named_modules())
    if target_name not in modules:
        candidates = [name for name in modules if name.endswith("self_attn.o_proj")]
        raise KeyError(f"Target {target_name!r} not found. Last candidates: {candidates[-5:]}")

    target = modules[target_name]
    if not isinstance(target, torch.nn.Linear):
        raise TypeError(f"Expected Linear, got {type(target)}")

    captured = {}

    def hook(_, inputs, output):
        captured["input"] = inputs[0].detach()
        if not output.requires_grad:
            output.requires_grad_(True)
        captured["output"] = output
        return output

    handle = target.register_forward_hook(hook)

    images = prepare_images_for_vla([image, wrist_image], cfg)
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    primary_batch = processor(
        prompt, images[0], return_tensors="pt"
    ).to(device, dtype=torch.bfloat16)
    wrist_batch = processor(
        prompt, images[1], return_tensors="pt"
    ).to(device, dtype=torch.bfloat16)

    primary_pixels = primary_batch["pixel_values"]
    wrist_pixels = wrist_batch["pixel_values"]
    combined_pixels = torch.cat(
        [primary_pixels, wrist_pixels],
        dim=1,
    ).contiguous()

    expected_channels = 6 * cfg.num_images_in_input

    print("primary pixel shape:", tuple(primary_pixels.shape))
    print("wrist pixel shape:", tuple(wrist_pixels.shape))
    print("combined pixel shape:", tuple(combined_pixels.shape))
    print("expected pixel channels:", expected_channels)
    print(
        "model num images:",
        model.vision_backbone.get_num_images_in_input(),
    )

    if combined_pixels.shape[1] != expected_channels:
        raise RuntimeError(
            f"Expected {expected_channels} pixel channels, "
            f"got {combined_pixels.shape[1]}"
        )

    model_inputs = {
        "input_ids": primary_batch["input_ids"],
        "attention_mask": primary_batch["attention_mask"],
        "pixel_values": combined_pixels,
    }

    proprio_stats = model.norm_stats[cfg.unnorm_key]["proprio"]
    normalized_state = normalize_proprio(state.copy(), proprio_stats)

    _, action_hidden_states = model.predict_action(
        **model_inputs,
        unnorm_key=cfg.unnorm_key,
        do_sample=False,
        proprio=normalized_state,
        proprio_projector=proprio_projector,
        action_head=action_head,
        noisy_action_projector=None,
        use_film=False,
    )

    handle.remove()

    normalized_actions = action_head.predict_action(action_hidden_states)
    physical_actions = differentiable_unnormalize(
        model, cfg.unnorm_key, normalized_actions
    )

    if "input" not in captured or "output" not in captured:
        channels = target.out_features
        zero_scores = {
            f"{mode}_{bits}": torch.zeros(channels, dtype=torch.float32)
            for mode in ("symmetric", "asymmetric")
            for bits in (2, 4, 8)
        }

        result = {
            "target": target_name,
            "instruction": instruction,
            "active": False,
            "probes": 0,
            "scores": zero_scores,
            "predicted_actions": physical_actions.detach().cpu(),
            "dataset_action": torch.from_numpy(dataset_action),
        }

        inactive_output = Path(output_path)
        inactive_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, inactive_output)

        print("target:", target_name)
        print("active: False")
        print("reason: target module was not called during VLA forward")
        print("channels:", channels)
        print("saved:", inactive_output)
        print("INACTIVE TARGET DETECTION: PASS")
        return

    layer_input = captured["input"].float()
    layer_output = captured["output"]
    weight = target.weight.detach().float()

    print("target:", target_name)
    print("layer input:", tuple(layer_input.shape))
    print("layer output:", tuple(layer_output.shape))
    print("predicted actions:", tuple(physical_actions.shape))
    print("predicted first action:", physical_actions[0, 0].detach().cpu().tolist())
    print("dataset first action:", dataset_action.tolist())

    deltas = {}
    for mode, quantizer in [
        ("symmetric", quantize_rows_symmetric),
        ("asymmetric", quantize_rows_asymmetric),
    ]:
        for bits in (2, 4, 8):
            quantized_weight = quantizer(weight, bits)
            weight_error = quantized_weight - weight
            deltas[f"{mode}_{bits}"] = F.linear(layer_input, weight_error)
            del quantized_weight, weight_error

    probes = 4
    generator = torch.Generator(device=device)
    generator.manual_seed(7)

    scores = {
        name: torch.zeros(target.out_features, device=device, dtype=torch.float32)
        for name in deltas
    }

    for probe_index in range(probes):
        random_vector = torch.randint(
            0,
            2,
            physical_actions.shape,
            generator=generator,
            device=device,
            dtype=torch.int64,
        ).float()
        random_vector = random_vector * 2.0 - 1.0

        objective = (physical_actions.float() * random_vector).sum()
        gradient = torch.autograd.grad(
            objective,
            layer_output,
            retain_graph=probe_index < probes - 1,
        )[0].float()

        reduce_dims = tuple(range(gradient.ndim - 1))

        for name, delta in deltas.items():
            projected = (gradient * delta).sum(dim=reduce_dims)
            scores[name] += projected.square() / probes

    cpu_scores = {}
    for name, score in sorted(scores.items()):
        score = score.detach().cpu()
        assert score.shape == (target.out_features,)
        assert torch.isfinite(score).all()
        assert (score >= 0).all()

        cpu_scores[name] = score
        print(
            name,
            "min=", float(score.min()),
            "mean=", float(score.mean()),
            "max=", float(score.max()),
            "nonzero=", int((score > 0).sum()),
        )

    result = {
        "target": target_name,
        "instruction": instruction,
        "active": True,
        "probes": probes,
        "scores": cpu_scores,
        "predicted_actions": physical_actions.detach().cpu(),
        "dataset_action": torch.from_numpy(dataset_action),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)

    print("saved:", output_path)
    print("ACTION-JACOBIAN SINGLE-LAYER SMOKE: PASS")


if __name__ == "__main__":
    main()
