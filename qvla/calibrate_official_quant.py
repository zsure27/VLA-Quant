from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qvla.action_jacobian_batch import initialize, load_sample, prepare_inputs
from qvla.official_quant_adapter import (
    AWQ_GROUP_SIZE,
    PROFILE_FORMAT,
    calibrate_awq_entry,
    input_absmax,
    load_official_functions,
    module_rows,
    read_targets,
    scope_counts,
    source_sha256,
    validate_targets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate official AWQ W4A16 or SmoothQuant W4A4 for OpenVLA."
    )
    parser.add_argument("--method", required=True, choices=("awq", "smoothquant"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples-dir", required=True, type=Path)
    parser.add_argument("--targets-file", required=True, type=Path)
    parser.add_argument("--official-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--bits", type=int, default=4, choices=(4,))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-rows", type=int, default=256)
    parser.add_argument("--max-output-rows", type=int, default=256)
    parser.add_argument("--clip-tokens", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=AWQ_GROUP_SIZE)
    parser.add_argument("--smooth-alpha", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    official = load_official_functions(args.official_root)
    names = read_targets(args.targets_file)
    counts = scope_counts(names)
    if counts["unexpected"]:
        raise RuntimeError(f"Unexpected target scope: {counts}")

    samples = sorted(args.samples_dir.glob("sample-*.npz"))[: args.num_samples]
    if len(samples) != args.num_samples:
        raise RuntimeError(f"Requested {args.num_samples} samples, found {len(samples)}")

    cfg, model, action_head, proprio_projector, processor = initialize(
        args.checkpoint, args.seed
    )
    selected = validate_targets(model, names)
    device = next(model.parameters()).device
    # 每次模型调用先取固定配额，最后再统一下采样。视觉层每个样本会由
    # 主相机和腕部相机各调用一次，因此不能在前 16 个样本就耗尽全局预算。
    rows_per_call = max(1, args.max_rows // args.num_samples)
    rows: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    capture_calls = {name: 0 for name in names}
    maxima: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        module = selected[name]

        def hook(_module: torch.nn.Module, values: tuple[Any, ...], _output: Any):
            value = values[0]
            capture_calls[name] += 1
            current_max = input_absmax(module, value).cpu()
            maxima[name] = (
                current_max
                if name not in maxima
                else torch.maximum(maxima[name], current_max)
            )
            if args.method == "awq":
                current = module_rows(module, value)
                take = min(rows_per_call, current.shape[0])
                if take:
                    index = torch.linspace(0, current.shape[0] - 1, take, device=current.device)
                    chosen = current[index.round().long()].to(torch.bfloat16).cpu()
                    rows[name].append(chosen)

        return hook

    for name, module in selected.items():
        handles.append(module.register_forward_hook(make_hook(name)))
    try:
        for index, path in enumerate(samples):
            sample = load_sample(path)
            model_inputs, state = prepare_inputs(sample, cfg, model, processor, device)
            with torch.no_grad():
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
            print(f"[capture] {index + 1}/{len(samples)} {path.name}")
    finally:
        for handle in handles:
            handle.remove()

    missing = [name for name in names if name not in maxima]
    if missing:
        raise RuntimeError(f"Targets were not executed: {missing}")

    payload: dict[str, Any] = {
        "format_version": PROFILE_FORMAT,
        "method": args.method,
        "bits": args.bits,
        "activation_bits": 16 if args.method == "awq" else 4,
        "checkpoint": str(args.checkpoint),
        "num_samples": args.num_samples,
        "sample_names": [path.name for path in samples],
        "scope": counts,
        "targets": names,
        "entries": {},
        "official_sources": {
            "awq_quantizer": {
                "path": str(official["awq_parent"] / "awq/quantize/quantizer.py"),
                "sha256": source_sha256(official["awq_parent"] / "awq/quantize/quantizer.py"),
            },
            "smoothquant_smooth": {
                "path": str(official["smoothquant_parent"] / "smoothquant/smooth.py"),
                "sha256": source_sha256(official["smoothquant_parent"] / "smoothquant/smooth.py"),
            },
            "smoothquant_fake_quant": {
                "path": str(official["smoothquant_parent"] / "smoothquant/fake_quant.py"),
                "sha256": source_sha256(official["smoothquant_parent"] / "smoothquant/fake_quant.py"),
            },
        },
        "adapter_notes": {
            "openvla_scope": "422 causally connected vision/language targets",
            "conv2d": "Patch embedding Conv2d is flattened for weight quantization",
            "awq": "Official asymmetric group fake quantizer, 20-point scale search, and auto clipping",
            "smoothquant": "Official smoothing and fake-quant primitives with n_bits=4",
        },
    }

    errors = []
    for index, name in enumerate(names):
        module = selected[name]
        if args.method == "smoothquant":
            entry = {
                "module_type": type(module).__name__,
                "shape": list(module.weight.shape),
                "activation_absmax": maxima[name].float(),
                "capture_calls": capture_calls[name],
            }
        else:
            calibration_rows = torch.cat(rows[name], dim=0)
            if calibration_rows.shape[0] > args.max_rows:
                index_rows = torch.linspace(
                    0, calibration_rows.shape[0] - 1, args.max_rows
                ).round().long().unique()
                calibration_rows = calibration_rows[index_rows]
            calibration_rows = calibration_rows.float().to(device)
            entry = calibrate_awq_entry(
                name,
                module,
                calibration_rows,
                official,
                bits=args.bits,
                group_size=args.group_size,
                max_output_rows=args.max_output_rows,
                clip_tokens=args.clip_tokens,
            )
            errors.append(entry["relative_mse"])
            entry["capture_calls"] = capture_calls[name]
            entry["calibration_rows"] = calibration_rows.shape[0]
            del calibration_rows
            torch.cuda.empty_cache()
        payload["entries"][name] = entry
        print(f"[target] {index + 1}/{len(names)} {name}")

    payload["group_size"] = args.group_size
    payload["smooth_alpha"] = args.smooth_alpha
    if errors:
        payload["mean_relative_mse"] = sum(errors) / len(errors)

    summary = {
        "format_version": PROFILE_FORMAT,
        "method": args.method,
        "weight_bits": args.bits,
        "activation_bits": payload["activation_bits"],
        "num_samples": args.num_samples,
        "scope": counts,
        "entries": len(payload["entries"]),
        "mean_relative_mse": payload.get("mean_relative_mse"),
        "output": str(args.output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    args.summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("OFFICIAL QUANTIZATION CALIBRATION: PASS")


if __name__ == "__main__":
    main()
