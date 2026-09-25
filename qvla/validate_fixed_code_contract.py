"""Validate the P1 fixed-code Scale-PEFT contract on backbone blocks 18-19."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from qvla.fixed_code_scale import FixedCodeScaleLinear, build_fixed_code, dequantize_fixed_code
from qvla.official_quant_adapter import load_official_functions, source_sha256
from qvla.run_eval_official_quant import load_profiles


def tensor_sha(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--official-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    from qvla.runtime_contract import assert_oft_runtime
    assert_oft_runtime()
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    entries, metadata = load_profiles([args.profile], "awq")
    official = load_official_functions(args.official_root)
    device = torch.device("cuda")
    model = OpenVLAForActionPrediction.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation=metadata["llm_attention"],
    ).to(device).eval()
    from qvla.awq_block import apply_block_scales
    modules = dict(model.named_modules())
    for prefix, scales in metadata["block_scales"].items():
        if prefix.endswith(".18") or prefix.endswith(".19"):
            apply_block_scales(modules[prefix], scales, official)

    rows = []
    wrappers = []
    for name, entry in entries.items():
        if not (name.startswith("language_model.model.layers.18.") or name.startswith("language_model.model.layers.19.")):
            continue
        layer = modules[name]
        weight = layer.weight.detach().clone()
        clip = entry.get("clip_max")
        if clip is not None:
            clip = clip.to(weight.device, weight.dtype)
            weight = weight.clone()
            weight.reshape(*clip.shape[:2], -1).clamp_(-clip, clip)
        official_value = official["awq_quantize"](weight, n_bit=2, zero_point=True, q_group_size=64)
        code = build_fixed_code(weight, 2, 64)
        rebuilt = dequantize_fixed_code(code)
        exact = torch.equal(official_value, rebuilt)
        max_abs = float((official_value.float() - rebuilt.float()).abs().max().item())
        rows.append({"name": name, "shape": list(weight.shape), "exact": exact, "max_abs": max_abs,
                     "q_sha256": tensor_sha(code.q), "zero_sha256": tensor_sha(code.zero),
                     "step_sha256": tensor_sha(code.step)})
        if not exact:
            raise RuntimeError(f"zero residual mismatch: {name}, max_abs={max_abs}")
        probe = torch.nn.Linear(layer.in_features, layer.out_features, bias=layer.bias is not None,
                                device=device, dtype=torch.bfloat16)
        probe.weight.data.copy_(weight)
        if layer.bias is not None:
            probe.bias.data.copy_(layer.bias)
        wrappers.append((name, FixedCodeScaleLinear(probe, 2, 64).to(device)))

    if len(rows) != 14:
        raise RuntimeError(f"expected 14 target linears in blocks 18-19, got {len(rows)}")
    optimizer = torch.optim.AdamW([m.log_step_residual for _, m in wrappers], lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.zeros((), device=device)
    for _name, module in wrappers:
        sample = torch.randn(2, module.in_features, device=device, dtype=torch.bfloat16)
        loss = loss + module(sample).float().square().mean()
    loss.backward()
    gradients = [m.log_step_residual.grad for _, m in wrappers]
    finite = all(g is not None and torch.isfinite(g).all() for g in gradients)
    nonzero = all(int(torch.count_nonzero(g).item()) > 0 for g in gradients)
    optimizer.step()
    state_path = args.output / "fixed-code-scale-state.pt"
    torch.save({name: module.state_dict() for name, module in wrappers}, state_path)
    reloaded = []
    for name, module in wrappers:
        clone = FixedCodeScaleLinear(torch.nn.Linear(module.in_features, module.out_features, bias=module.bias is not None,
                                                      device=device, dtype=torch.bfloat16), 2, 64).to(device)
        clone.load_state_dict(torch.load(state_path, map_location=device, weights_only=True)[name])
        reloaded.append(torch.equal(module.q, clone.q) and torch.equal(module.zero, clone.zero)
                        and torch.equal(module.step, clone.step)
                        and torch.equal(module.log_step_residual, clone.log_step_residual))
    result = {"status": "PASS" if finite and nonzero and all(reloaded) else "FAIL",
              "targets": rows, "target_count": len(rows), "loss": float(loss.detach().item()),
              "finite_gradients": finite, "nonzero_gradients": nonzero,
              "reload_exact": all(reloaded), "trainable_parameters": sum(m.log_step_residual.numel() for _, m in wrappers),
              "profile_sha256": source_sha256(args.profile), "checkpoint": str(args.checkpoint)}
    (args.output / "contract.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "targets"}, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
