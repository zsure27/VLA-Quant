"""Frozen quantized Linear plus a high-precision low-rank residual branch.

SVD initialization is a weight reconstruction probe, not calibrated training.
Attach this branch and its hook before loading a saved state dictionary.
"""
import math
import torch
from torch import nn

def attach_residual(linear, teacher_weight, rank, seed, input_rms=None, input_rows=None,
                    init="response-svd", train_steps=0, learning_rate=1e-3):
    if not isinstance(linear, nn.Linear) or rank<1 or rank>=min(linear.weight.shape):
        raise ValueError("low-rank residual requires a Linear and a valid rank")
    if hasattr(linear,"awq_recovery_lora"):
        raise ValueError("residual branch already attached")
    residual=teacher_weight.float()-linear.weight.detach().float()
    if input_rows is not None and input_rms is not None: raise ValueError("choose RMS or response initialization")
    x=None if input_rows is None else input_rows.to(residual.device).float()
    if x is not None and (x.ndim!=2 or x.shape[1]!=residual.shape[1] or x.shape[0]<rank or not torch.isfinite(x).all()):
        raise ValueError("response initialization requires finite input rows with matching channels")
    scale=torch.ones(residual.shape[1],device=residual.device) if input_rms is None else input_rms.to(residual.device).float()
    if scale.shape != (residual.shape[1],) or not torch.isfinite(scale).all() or (scale<0).any():
        raise ValueError("input RMS requires finite nonnegative per-channel statistics")
    scale=scale.clamp_min(1e-4)
    devices=[linear.weight.device.index] if linear.weight.is_cuda else []
    if init not in ("response-svd", "standard-zero"):
        raise ValueError("unknown Recovery LoRA initialization")
    if train_steps < 0 or (train_steps and x is None):
        raise ValueError("training requires captured input rows")
    with torch.no_grad(),torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if init == "response-svd":
            objective=residual*scale if x is None else residual@x.T
            u,s,v=torch.svd_lowrank(objective,q=min(rank+8,min(objective.shape)),niter=2)
            if x is None:
                a=v[:,:rank].T/scale
                b=u[:,:rank]*s[:rank]
            else:
                b=u[:,:rank]
                a=b.T@residual
        else:
            objective=residual@x.T
            a=torch.empty(rank,residual.shape[1],device=residual.device)
            torch.nn.init.kaiming_uniform_(a,a=math.sqrt(5))
            b=torch.zeros(residual.shape[0],rank,device=residual.device)
        branch=nn.Sequential(nn.Linear(linear.in_features,rank,bias=False),
                             nn.Linear(rank,linear.out_features,bias=False)).to(linear.weight.device,linear.weight.dtype)
        branch[0].weight.copy_(a.to(branch[0].weight))
        branch[1].weight.copy_(b.to(branch[1].weight))
        energy=residual.square().sum().clamp_min(1e-30)
        unexplained=(residual-b@a).square().sum()/energy
        weighted_unexplained=((residual-b@a)*scale).square().sum()/(residual*scale).square().sum().clamp_min(1e-30)
        response_unexplained=None if x is None else float(((residual-b@a)@x.T).square().sum()/objective.square().sum().clamp_min(1e-30))
        initial_output_exact_zero=bool(torch.count_nonzero(branch[1].weight).item()==0)
    initial_loss=None
    final_loss=None
    if train_steps:
        target=(x@residual.T).detach()
        optimizer=torch.optim.AdamW(branch.parameters(),lr=learning_rate)
        with torch.enable_grad():
            for step_index in range(train_steps):
                optimizer.zero_grad(set_to_none=True)
                prediction=branch(x.to(branch[0].weight.dtype)).float()
                loss=(prediction-target).square().mean()
                if step_index==0: initial_loss=float(loss.detach())
                if not torch.isfinite(loss): raise RuntimeError("non-finite Recovery LoRA loss")
                loss.backward()
                if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters()):
                    raise RuntimeError("invalid Recovery LoRA gradient")
                optimizer.step()
            final_loss=float(loss.detach())
    for p in linear.parameters(): p.requires_grad_(False)
    linear.add_module("awq_recovery_lora",branch)
    def correction(module,args,output):
        return output+module.awq_recovery_lora(args[0])
    linear.register_forward_hook(correction)
    return {"rank":rank,"adapter_parameters":sum(p.numel() for p in branch.parameters()),
            "residual_frobenius_unexplained_fraction":float(unexplained),
            "input_diagonal_unexplained_fraction":float(weighted_unexplained),
            "response_unexplained_fraction":response_unexplained,
            "init":init,"seed":seed,"initial_output_exact_zero":initial_output_exact_zero,
            "training_steps":train_steps,"learning_rate":learning_rate,
            "training_initial_mse":initial_loss,"training_final_mse":final_loss}
