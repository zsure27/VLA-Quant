"""Attach and train fixed-code Scale-PEFT corrections for an AWQ Linear."""
from __future__ import annotations

import hashlib

import torch
import torch.nn.functional as F
from torch import nn

from qvla.fixed_code_scale import FixedCodeTensors, build_fixed_code, dequantize_fixed_code


def _sha(value: torch.Tensor) -> str:
    raw=value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


class FixedCodeScaleCorrection(nn.Module):
    def __init__(self, code: FixedCodeTensors):
        super().__init__()
        self.shape=code.shape; self.group_size=code.group_size; self.bits=code.bits
        self.register_buffer("q",code.q,persistent=True)
        self.register_buffer("zero",code.zero,persistent=True)
        self.register_buffer("step",code.step,persistent=True)
        self.log_step_residual=nn.Parameter(torch.zeros_like(code.step,dtype=torch.float32))

    def code(self):
        return FixedCodeTensors(self.q,self.zero,self.step,self.shape,self.group_size,self.bits)

    def forward(self,value):
        code=self.code()
        correction=dequantize_fixed_code(code,self.log_step_residual)-dequantize_fixed_code(code)
        return F.linear(value,correction)


def attach_scale_peft(linear,teacher_weight,entry,bits,group_size,input_rows,train_steps,learning_rate):
    if not isinstance(linear,nn.Linear) or hasattr(linear,"awq_scale_peft"):
        raise ValueError("Scale-PEFT requires an unattached Linear")
    prepared=teacher_weight.detach().clone()
    clip=entry.get("clip_max")
    if clip is not None:
        clip=clip.to(prepared.device,prepared.dtype)
        prepared.reshape(*clip.shape[:2],-1).clamp_(-clip,clip)
    code=build_fixed_code(prepared,bits,group_size)
    rebuilt=dequantize_fixed_code(code)
    if not torch.equal(rebuilt,linear.weight.detach()):
        raise RuntimeError("fixed-code zero residual does not reproduce official AWQ weight")
    branch=FixedCodeScaleCorrection(code).to(linear.weight.device)
    x=input_rows.to(linear.weight.device).float()
    target=(x@(teacher_weight.detach().float()-rebuilt.float()).T).detach()
    optimizer=torch.optim.AdamW(branch.parameters(),lr=learning_rate) if train_steps else None
    initial_loss=final_loss=None
    with torch.enable_grad():
        for index in range(train_steps):
            optimizer.zero_grad(set_to_none=True)
            prediction=branch(x.to(linear.weight.dtype)).float()
            loss=(prediction-target).square().mean()
            if index==0: initial_loss=float(loss.detach())
            if not torch.isfinite(loss): raise RuntimeError("non-finite Scale-PEFT loss")
            loss.backward()
            grad=branch.log_step_residual.grad
            if grad is None or not torch.isfinite(grad).all(): raise RuntimeError("invalid Scale-PEFT gradient")
            optimizer.step()
        if train_steps:
            final_loss=float(loss.detach())
    for parameter in linear.parameters(): parameter.requires_grad_(False)
    linear.add_module("awq_scale_peft",branch)
    def correction(module,args,output): return output+module.awq_scale_peft(args[0])
    linear.register_forward_hook(correction)
    return {"trainable_parameters":branch.log_step_residual.numel(),"training_steps":train_steps,
            "learning_rate":learning_rate,"training_initial_mse":initial_loss,"training_final_mse":final_loss,
            "q_sha256":_sha(code.q),"zero_sha256":_sha(code.zero),"step_sha256":_sha(code.step),
            "zero_residual_exact":True,"max_abs_log_step_residual":float(branch.log_step_residual.detach().abs().max())}
