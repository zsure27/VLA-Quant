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

