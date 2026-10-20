"""Small capability registry. Adding a name never asserts numerical compatibility."""
from __future__ import annotations

SUITES = {name: {"tasks": 10, "evaluation_only": False} for name in
          ("libero_spatial", "libero_object", "libero_goal", "libero_10")}
QUANTIZERS = {
    "awq": {"backend": "official_quant", "weight_bits": [2, 4, 8], "activation_bits": [16]},
    "smoothquant": {"backend": "official_quant", "weight_bits": [2, 4, 8], "activation_bits": [4, 8, 16]},
    "none": {"backend": "oft_bf16", "weight_bits": [16], "activation_bits": [16]},
}
STRATEGIES = {
    "none": {"binding_required": False},
    "lora": {"binding_required": True, "historical_entry": "scripts/train_extended_peft.py"},
    "response_svd_lora": {"binding_required": True, "historical_entry": "scripts/train_b3_initialization_v1.py"},
    "routing": {"binding_required": True, "historical_entry": "scripts/train_b5_router.py"},
}


def register_suite(name, tasks):
    if name in SUITES or not isinstance(tasks, int) or tasks < 1:
        raise ValueError("Duplicate suite or invalid task count")
    SUITES[name] = {"tasks": tasks, "evaluation_only": False}


def register_quantizer(name, weight_bits, activation_bits):
    if name in QUANTIZERS:
        raise ValueError("Duplicate quantizer")
    QUANTIZERS[name] = {"backend": "command", "weight_bits": list(weight_bits),
                        "activation_bits": list(activation_bits)}


def register_strategy(name):
    if name in STRATEGIES:
        raise ValueError("Duplicate strategy")
    STRATEGIES[name] = {"binding_required": True}
