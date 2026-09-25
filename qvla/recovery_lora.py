"""Minimal Recovery LoRA used by the blocks 18-19 P1/P2 experiments."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class RecoveryLoRALinear(torch.nn.Module):
    def __init__(self, source: torch.nn.Linear, rank: int = 8, alpha: float = 8.0):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / rank
        self.base_weight = torch.nn.Parameter(source.weight.detach().clone(), requires_grad=False)
        if source.bias is None:
            self.register_parameter("base_bias", None)
        else:
            self.base_bias = torch.nn.Parameter(source.bias.detach().clone(), requires_grad=False)
        self.lora_A = torch.nn.Parameter(torch.empty(rank, source.in_features, device=source.weight.device,
                                                     dtype=source.weight.dtype))
        self.lora_B = torch.nn.Parameter(torch.zeros(source.out_features, rank, device=source.weight.device,
                                                      dtype=source.weight.dtype))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base = F.linear(value, self.base_weight, self.base_bias)
        delta = F.linear(F.linear(value, self.lora_A), self.lora_B) * self.scaling
        return base + delta


@torch.no_grad()
def attach_recovery_lora_state(
    linear: torch.nn.Linear,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> dict[str, int]:
    """Attach the exact unscaled two-linear branch saved by diagnostics."""
    if not isinstance(linear, torch.nn.Linear) or hasattr(linear, "awq_recovery_lora"):
        raise ValueError("Recovery LoRA state requires a bare Linear")
    if down_weight.ndim != 2 or up_weight.ndim != 2:
        raise ValueError("Recovery LoRA state tensors must be matrices")
    rank = down_weight.shape[0]
    if tuple(down_weight.shape) != (rank, linear.in_features):
        raise ValueError("Recovery LoRA down projection shape mismatch")
    if tuple(up_weight.shape) != (linear.out_features, rank):
        raise ValueError("Recovery LoRA up projection shape mismatch")
    if rank < 1 or not torch.isfinite(down_weight).all() or not torch.isfinite(up_weight).all():
        raise ValueError("Recovery LoRA state is invalid")
    branch = torch.nn.Sequential(
        torch.nn.Linear(linear.in_features, rank, bias=False),
        torch.nn.Linear(rank, linear.out_features, bias=False),
    ).to(device=linear.weight.device, dtype=linear.weight.dtype)
    branch[0].weight.copy_(down_weight.to(branch[0].weight))
    branch[1].weight.copy_(up_weight.to(branch[1].weight))
    branch.requires_grad_(False)
    linear.add_module("awq_recovery_lora", branch)

    def correction(module, args, output):
        return output + module.awq_recovery_lora(args[0])

    linear.register_forward_hook(correction)
    return {"rank": rank, "parameters": down_weight.numel() + up_weight.numel()}
