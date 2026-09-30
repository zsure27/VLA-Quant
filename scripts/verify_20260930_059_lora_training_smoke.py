#!/usr/bin/env python3
"""Fail-closed structural audit for the 10-step wide-language-LoRA smoke."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

import torch

ELIGIBLE = set(range(0, 8)) | set(range(16, 20)) | set(range(24, 32))
FAMILIES = {
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
}
KEY = re.compile(
    r"^language_model\.model\.layers\.(\d+)\.(self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))\.awq_recovery_lora\.([01])\.weight$"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    scope = json.loads((output / "scope.json").read_text())
    distill = json.loads((output / "e2e_distillation.json").read_text())
    rows = scope.get("low_rank_residual", {})
    expected_modules = {
        f"language_model.model.layers.{layer}.{family}"
        for layer in ELIGIBLE for family in FAMILIES
    }
    if set(rows) != expected_modules:
        raise SystemExit(f"target module mismatch: expected {len(expected_modules)}, got {len(rows)}")
    if any(row.get("rank") != 8 or row.get("init") != "response-svd" for row in rows.values()):
        raise SystemExit("rank or Response-SVD initialization mismatch")
    if any(not math.isfinite(float(row["response_unexplained_fraction"]))
           for row in rows.values() if row.get("response_unexplained_fraction") is not None):
        raise SystemExit("non-finite response-SVD diagnostic")
    state_path = output / "e2e_adapter_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or len(state) != len(expected_modules) * 2:
        raise SystemExit("saved state tensor count mismatch")
    parsed: dict[tuple[int, str], set[int]] = {}
    for name, tensor in state.items():
        match = KEY.fullmatch(name)
        if not match or not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            raise SystemExit(f"invalid adapter state entry: {name}")
        layer, family, factor = int(match.group(1)), match.group(2), int(match.group(3))
        if layer not in ELIGIBLE or tensor.ndim != 2 or not torch.isfinite(tensor).all():
            raise SystemExit(f"invalid target/tensor: {name}")
        if (layer, family) not in parsed:
            parsed[(layer, family)] = set()
        parsed[(layer, family)].add(factor)
    if set(parsed) != {(int(re.search(r"layers\.(\d+)", n).group(1)), f)
                        for n in expected_modules for f in FAMILIES} or any(v != {0, 1} for v in parsed.values()):
        raise SystemExit("adapter coverage is incomplete")
    if distill.get("steps") != 10 or distill.get("loss") != "smooth-l1" or distill.get("smooth_l1_beta") != 0.1:
        raise SystemExit("smoke training protocol mismatch")
    if distill.get("trainable_parameters") != sum(t.numel() for t in state.values()):
        raise SystemExit("trainable-parameter count differs from saved state")
    if not math.isfinite(float(distill.get("initial_train_mse", float("nan")))) or not math.isfinite(float(distill.get("final_train_mse", float("nan")))):
        raise SystemExit("non-finite smoke loss")
    print(json.dumps({
        "status": "PASS_TRAINING_STRUCTURE_SMOKE",
        "target_blocks": sorted(ELIGIBLE),
        "target_linear_modules": len(expected_modules),
        "adapter_tensors": len(state),
        "trainable_parameters": int(distill["trainable_parameters"]),
        "initial_train_mse": distill["initial_train_mse"],
        "final_train_mse": distill["final_train_mse"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
