"""Fixed-code asymmetric group quantization for Scale-PEFT experiments.

The arithmetic intentionally mirrors the pinned AWQ ``pseudo_quantize_tensor``
path.  Integer codes and zero points stay frozen; only a log residual on the
dequantization step is trainable.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FixedCodeTensors:
    q: torch.Tensor
    zero: torch.Tensor
    step: torch.Tensor
    shape: tuple[int, ...]
    group_size: int
    bits: int


def build_fixed_code(weight: torch.Tensor, bits: int, group_size: int) -> FixedCodeTensors:
    if bits not in (2, 4, 8) or group_size < 1:
        raise ValueError("unsupported fixed-code quantization contract")
    if weight.shape[-1] % group_size:
        raise ValueError("the language contract does not permit implicit padding")
    rows = weight.detach().reshape(-1, group_size)
    max_value = rows.amax(dim=1, keepdim=True)
    min_value = rows.amin(dim=1, keepdim=True)
    max_int = 2**bits - 1
    step = (max_value - min_value).clamp(min=1e-5) / max_int
    zero = (-torch.round(min_value / step)).clamp_(0, max_int)
    q = torch.clamp(torch.round(rows / step) + zero, 0, max_int)
    return FixedCodeTensors(
        q=q.to(torch.uint8),
        zero=zero.to(torch.uint8),
        step=step.clone(),
        shape=tuple(weight.shape),
        group_size=group_size,
        bits=bits,
    )


def dequantize_fixed_code(code: FixedCodeTensors, log_step_residual: torch.Tensor | None = None) -> torch.Tensor:
    step = code.step
    if log_step_residual is not None:
        if log_step_residual.shape != step.shape:
            raise ValueError("log-step residual shape mismatch")
        # Casting before exp/multiply preserves the exact pinned BF16 zero path.
        residual = log_step_residual.to(device=step.device, dtype=step.dtype)
        step = step * torch.exp(residual)
    value = (code.q.to(step.dtype) - code.zero.to(step.dtype)) * step
    return value.reshape(code.shape)


def official_equivalent(weight: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    """Independent transcription of the pinned official fake-quant formula."""
    shape = weight.shape
    rows = weight.reshape(-1, group_size)
    max_value = rows.amax(dim=1, keepdim=True)
    min_value = rows.amin(dim=1, keepdim=True)
    max_int = 2**bits - 1
    step = (max_value - min_value).clamp(min=1e-5) / max_int
    zero = (-torch.round(min_value / step)).clamp_(0, max_int)
    value = (torch.clamp(torch.round(rows / step) + zero, 0, max_int) - zero) * step
    return value.reshape(shape)


class FixedCodeScaleLinear(torch.nn.Module):
    """Linear layer with frozen q/z/step and trainable per-group log-step residual."""

    def __init__(self, source: torch.nn.Linear, bits: int, group_size: int):
        super().__init__()
        code = build_fixed_code(source.weight.detach(), bits, group_size)
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.bits = bits
        self.group_size = group_size
        self.register_buffer("q", code.q, persistent=True)
        self.register_buffer("zero", code.zero, persistent=True)
        self.register_buffer("step", code.step, persistent=True)
        self.log_step_residual = torch.nn.Parameter(torch.zeros_like(code.step, dtype=torch.float32))
        if source.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = torch.nn.Parameter(source.bias.detach().clone(), requires_grad=False)

    def fixed_code(self) -> FixedCodeTensors:
        return FixedCodeTensors(self.q, self.zero, self.step, (self.out_features, self.in_features), self.group_size, self.bits)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weight = dequantize_fixed_code(self.fixed_code(), self.log_step_residual)
        return F.linear(value, weight, self.bias)


class FixedCodeScaleCorrection(torch.nn.Module):
    """Additive correction that preserves the frozen fake-quantized base weight."""

    def __init__(self, code: FixedCodeTensors):
        super().__init__()
        self.shape = code.shape
        self.group_size = code.group_size
        self.bits = code.bits
        self.register_buffer("q", code.q, persistent=True)
        self.register_buffer("zero", code.zero, persistent=True)
        self.register_buffer("step", code.step, persistent=True)
        self.log_step_residual = torch.nn.Parameter(
            torch.zeros_like(code.step, dtype=torch.float32), requires_grad=False
        )

    def fixed_code(self) -> FixedCodeTensors:
        return FixedCodeTensors(
            self.q, self.zero, self.step, self.shape, self.group_size, self.bits
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        code = self.fixed_code()
        correction = dequantize_fixed_code(code, self.log_step_residual) - dequantize_fixed_code(code)
        return F.linear(value, correction)


def attach_fixed_code_scale_state(
    linear: torch.nn.Linear,
    teacher_weight: torch.Tensor,
    entry: dict,
    bits: int,
    group_size: int,
    log_step_residual: torch.Tensor,
) -> dict:
    """Recreate a trained Scale-PEFT branch and attach its frozen residual state."""
    if not isinstance(linear, torch.nn.Linear) or hasattr(linear, "awq_scale_peft"):
        raise ValueError("Scale-PEFT state requires an unattached Linear")
    prepared = teacher_weight.detach().clone()
    clip = entry.get("clip_max")
    if clip is not None:
        clip = clip.to(prepared.device, prepared.dtype)
        prepared.reshape(*clip.shape[:2], -1).clamp_(-clip, clip)
    code = build_fixed_code(prepared, bits, group_size)
    rebuilt = dequantize_fixed_code(code)
    if not torch.equal(rebuilt, linear.weight.detach()):
        raise RuntimeError("Scale-PEFT state does not match the frozen AWQ base weight")
    branch = FixedCodeScaleCorrection(code).to(linear.weight.device)
    if tuple(log_step_residual.shape) != tuple(branch.log_step_residual.shape):
        raise RuntimeError("Scale-PEFT residual shape mismatch")
    if not torch.isfinite(log_step_residual).all():
        raise RuntimeError("Scale-PEFT residual contains non-finite values")
    branch.log_step_residual.copy_(
        log_step_residual.to(branch.log_step_residual.device, branch.log_step_residual.dtype)
    )
    linear.add_module("awq_scale_peft", branch)

    def correction(module, args, output):
        return output + module.awq_scale_peft(args[0])

    linear.register_forward_hook(correction)
    return {
        "parameters": branch.log_step_residual.numel(),
        "max_abs_log_step_residual": float(branch.log_step_residual.abs().max()),
        "zero_residual_exact": True,
    }
