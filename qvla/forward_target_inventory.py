from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model
from experiments.robot.openvla_utils import normalize_proprio, prepare_images_for_vla


def main():
    sample_path, checkpoint, proxy_path, output_path = sys.argv[1:5]
    device = torch.device("cuda:0")

    proxy = torch.load(proxy_path, map_location="cpu")
    target_names = list(proxy)

    with np.load(sample_path, allow_pickle=False) as sample:
        image = sample["image"]
        wrist_image = sample["wrist_image"]
        state = sample["state"].astype(np.float32)
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
    model.eval()
    action_head.eval()
    proprio_projector.eval()
    model.vision_backbone.set_num_images_in_input(2)

    modules = dict(model.named_modules())
    records = {}
    handles = []

    for name in target_names:
        module = modules.get(name)

        if module is None:
            records[name] = {
                "exists": False,
                "calls": 0,
                "type": None,
                "channels": 0,
                "weight_parameters": 0,
            }
            continue

        weight = getattr(module, "weight", None)
        channels = int(weight.shape[0]) if weight is not None else 0
        parameters = int(weight.numel()) if weight is not None else 0

        records[name] = {
            "exists": True,
            "calls": 0,
            "type": type(module).__name__,
            "channels": channels,
            "weight_parameters": parameters,
            "input_shape": None,
            "output_shape": None,
        }

        def make_hook(module_name):
            def hook(_, inputs, output):
                record = records[module_name]
                record["calls"] += 1

                if record["input_shape"] is None:
                    value = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
                    if torch.is_tensor(value):
                        record["input_shape"] = list(value.shape)

                    output_value = output
                    if isinstance(output_value, (tuple, list)):
                        output_value = output_value[0]
                    if torch.is_tensor(output_value):
                        record["output_shape"] = list(output_value.shape)

            return hook

        handles.append(module.register_forward_hook(make_hook(name)))

    images = prepare_images_for_vla([image, wrist_image], cfg)
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    primary = processor(
        prompt, images[0], return_tensors="pt"
    ).to(device, dtype=torch.bfloat16)
    wrist = processor(
        prompt, images[1], return_tensors="pt"
    ).to(device, dtype=torch.bfloat16)

    pixels = torch.cat(
        [primary["pixel_values"], wrist["pixel_values"]],
        dim=1,
    ).contiguous()

    assert pixels.shape[1] == 12, pixels.shape

    model_inputs = {
        "input_ids": primary["input_ids"],
        "attention_mask": primary["attention_mask"],
        "pixel_values": pixels,
    }

    proprio_stats = model.norm_stats[cfg.unnorm_key]["proprio"]
    normalized_state = normalize_proprio(state.copy(), proprio_stats)

    with torch.inference_mode():
        model.predict_action(
            **model_inputs,
            unnorm_key=cfg.unnorm_key,
            do_sample=False,
            proprio=normalized_state,
            proprio_projector=proprio_projector,
            action_head=action_head,
            noisy_actionprojector=None,
            use_film=False,
        )

    for handle in handles:
        handle.remove()

    existing = [name for name, record in records.items() if record["exists"]]
    called = [name for name, record in records.items() if record["calls"] > 0]
    uncalled = [
        name for name, record in records.items()
        if record["exists"] and record["calls"] == 0
    ]
    missing = [name for name, record in records.items() if not record["exists"]]

    summary = {
        "proxy_targets": len(target_names),
        "existing_targets": len(existing),
        "called_targets": len(called),
        "uncalled_targets": len(uncalled),
        "missing_targets": len(missing),
        "called_channels": sum(records[name]["channels"] for name in called),
        "uncalled_channels": sum(records[name]["channels"] for name in uncalled),
        "called_weight_parameters": sum(
            records[name]["weight_parameters"] for name in called
        ),
        "uncalled_weight_parameters": sum(
            records[name]["weight_parameters"] for name in uncalled
        ),
    }

    result = {
        "summary": summary,
        "called": called,
        "uncalled": uncalled,
        "missing": missing,
        "records": records,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(summary, indent=2))
    print("\nUncalled targets:")
    for name in uncalled:
        print(
            name,
            "channels=", records[name]["channels"],
            "type=", records[name]["type"],
        )

    print("\nMissing targets:")
    for name in missing:
        print(name)

    print("\nsaved:", output_path)
    print("FORWARD TARGET INVENTORY: PASS")


if __name__ == "__main__":
    main()
