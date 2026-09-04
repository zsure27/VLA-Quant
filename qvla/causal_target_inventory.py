import json
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model
from experiments.robot.openvla_utils import normalize_proprio, prepare_images_for_vla


def unnormalize(model, key, actions):
    stats = model.get_action_stats(key)

    if "q99" in stats:
        high, low = stats["q99"], stats["q01"]
    else:
        high, low = stats["max"], stats["min"]

    high = torch.as_tensor(high, device=actions.device, dtype=torch.float32)
    low = torch.as_tensor(low, device=actions.device, dtype=torch.float32)
    mask = torch.as_tensor(
        stats.get("mask", np.ones(len(high), dtype=bool)),
        device=actions.device,
        dtype=torch.bool,
    )

    physical = 0.5 * (actions.float() + 1) * (high - low + 1e-8) + low
    return torch.where(mask, physical, actions.float())


def main():
    sample_path, checkpoint, inventory_path, prefix, output_path = sys.argv[1:6]
    device = torch.device("cuda:0")

    inventory = json.loads(Path(inventory_path).read_text())
    candidates = [
        name for name in inventory["called"]
        if name.startswith(prefix)
    ]

    with np.load(sample_path, allow_pickle=False) as sample:
        image = sample["image"]
        wrist = sample["wrist_image"]
        state = sample["state"].astype(np.float32)
        instruction = str(sample["instruction"].item())

    cfg = GenerateConfig(
        pretrained_checkpoint=checkpoint,
        task_suite_name="libero_spatial",
        use_l1_regression=True,
        use_diffusion=False,
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
    captured = {name: [] for name in candidates}
    handles = []

    def make_hook(name):
        def hook(_, inputs, output):
            if not torch.is_tensor(output):
                raise TypeError(f"{name} returned {type(output)}")
            if not output.requires_grad:
                output.requires_grad_(True)
            captured[name].append(output)
            return output
        return hook

    for name in candidates:
        handles.append(
            modules[name].register_forward_hook(make_hook(name))
        )

    images = prepare_images_for_vla([image, wrist], cfg)
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    primary = processor(
        prompt, images[0], return_tensors="pt"
    ).to(device, dtype=torch.bfloat16)
    wrist_batch = processor(
        prompt, images[1], return_tensors="pt"
    ).to(device, dtype=torch.bfloat16)

    pixels = torch.cat(
        [primary["pixel_values"], wrist_batch["pixel_values"]],
        dim=1,
    ).contiguous()
    assert pixels.shape[1] == 12

    proprio_stats = model.norm_stats[cfg.unnorm_key]["proprio"]
    normalized_state = normalize_proprio(state.copy(), proprio_stats)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    _, hidden = model.predict_action(
        input_ids=primary["input_ids"],
        attention_mask=primary["attention_mask"],
        pixel_values=pixels,
        unnorm_key=cfg.unnorm_key,
        do_sample=False,
        proprio=normalized_state,
        proprio_projector=proprio_projector,
        action_head=action_head,
        noisy_action_projector=None,
        use_film=False,
    )

    for handle in handles:
        handle.remove()

    normalized_actions = action_head.predict_action(hidden)
    actions = unnormalize(model, cfg.unnorm_key, normalized_actions)

    generator = torch.Generator(device=device).manual_seed(7)
    probe = torch.randint(
        0, 2, actions.shape,
        generator=generator,
        device=device,
    ).float() * 2 - 1

    objective = (actions * probe).sum()

    tensors = []
    owners = []
    for name in candidates:
        for tensor in captured[name]:
            tensors.append(tensor)
            owners.append(name)

    gradients = torch.autograd.grad(
        objective,
        tensors,
        allow_unused=True,
    )

    records = {
        name: {
            "calls": len(captured[name]),
            "connected": False,
            "nonzero_gradient": False,
            "gradient_norm": 0.0,
        }
        for name in candidates
    }

    for name, gradient in zip(owners, gradients):
        if gradient is None:
            continue

        records[name]["connected"] = True
        norm = float(gradient.float().norm().detach().cpu())
        records[name]["gradient_norm"] += norm

        if norm > 0:
            records[name]["nonzero_gradient"] = True

    connected = [
        name for name in candidates
        if records[name]["connected"]
    ]
    disconnected = [
        name for name in candidates
        if not records[name]["connected"]
    ]
    zero_gradient = [
        name for name in connected
        if not records[name]["nonzero_gradient"]
    ]

    summary = {
        "prefix": prefix,
        "candidates": len(candidates),
        "connected": len(connected),
        "disconnected": len(disconnected),
        "zero_gradient": len(zero_gradient),
        "captured_calls": len(tensors),
        "peak_gpu_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30
        ),
        "peak_gpu_reserved_gib": (
            torch.cuda.max_memory_reserved(device) / 2**30
        ),
    }

    result = {
        "summary": summary,
        "connected": connected,
        "disconnected": disconnected,
        "zero_gradient": zero_gradient,
        "records": records,
    }

    Path(output_path).write_text(json.dumps(result, indent=2))

    print(json.dumps(summary, indent=2))
    print("\nDisconnected:")
    for name in disconnected:
        print(name)

    print("\nZero-gradient but graph-connected:")
    for name in zero_gradient:
        print(name)

    print("\nsaved:", output_path)
    print("CAUSAL TARGET INVENTORY: PASS")


if __name__ == "__main__":
    main()
