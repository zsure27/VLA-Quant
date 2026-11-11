"""Pinned SmoothQuant quantizers plus explicit Gemma/adaRMS affine folding.

414 attention/MLP Linear targets; embedding, patch convolution, normalization,
conditioning and action I/O projections retain the official BF16/FP32 policy.
This is a model-specific binding, not an INT4 execution kernel.
"""
import hashlib
import importlib.util
from pathlib import Path

import torch

from qvla.pi05_torch_backend_v1 import target_names


def official_functions(root):
    candidates=list((Path(root)/'sources/quant-source-sparse-v2').rglob('fake_quant.py'))
    expected='d0979f66c1bc38f4146db8365a162c5b356401f864636076395a978a9d569787'
    matches=[p for p in candidates if hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==expected]
    if len(matches)!=1:raise ValueError('Pinned SmoothQuant source is missing or ambiguous')
    path=matches[0];spec=importlib.util.spec_from_file_location('_pi05_pinned_sq_fake',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,path


def groups(model):
    modules=dict(model.named_modules());names=set(target_names(model));result=[]
    for name in sorted(names):
        if name.endswith('.self_attn.q_proj'):
            prefix=name[:-len('.self_attn.q_proj')]
            norm=prefix+('.layer_norm1' if '.vision_tower.' in name else '.input_layernorm')
            targets=[prefix+'.self_attn.'+x for x in ['q_proj','k_proj','v_proj']]
            result.append((norm,targets))
        elif name.endswith('.mlp.gate_proj'):
            prefix=name[:-len('.mlp.gate_proj')]
            result.append((prefix+'.post_attention_layernorm',[prefix+'.mlp.gate_proj',prefix+'.mlp.up_proj']))
        elif name.endswith('.mlp.fc1'):
            prefix=name[:-len('.mlp.fc1')]
            result.append((prefix+'.layer_norm2',[name]))
    if len(names)!=414 or len(result)!=126:raise ValueError('Unexpected pi0.5 smoothing scope')
    if any(n not in modules or any(x not in names for x in targets) for n,targets in result):
        raise ValueError('Missing smoothing predecessor/target')
    return result


@torch.no_grad()
def fold_norm(norm, scale):
    """Transform norm output to y/scale while preserving any adaptive gate."""
    if scale.ndim!=1 or not torch.isfinite(scale).all() or (scale<=0).any():
        raise ValueError('Invalid positive channel scale')
    if isinstance(norm,torch.nn.LayerNorm):
        norm.weight.div_(scale.to(norm.weight))
        if norm.bias is not None:norm.bias.div_(scale.to(norm.bias))
    elif type(norm).__name__=='GemmaRMSNorm':
        dense=getattr(norm,'dense',None)
        if dense is None:
            s=scale.to(norm.weight)
            norm.weight.copy_((1+norm.weight)/s-1)
        else:
            dim=scale.numel()
            if dense.weight.shape[0]!=3*dim or dense.bias is None:
                raise ValueError('Unexpected adaptive scale/shift/gate layout')
            s=scale.to(dense.weight)
            # scale(c)+1 and shift(c) are divided; gate(c) is unchanged.
            dense.weight[:2*dim].div_(s.repeat(2)[:,None])
            dense.bias[:dim].copy_((1+dense.bias[:dim])/s-1)
            dense.bias[dim:2*dim].div_(s)
    else:raise TypeError('Unaudited smoothing norm '+type(norm).__name__)


@torch.no_grad()
def smooth(model, stats, alpha=0.5):
    modules=dict(model.named_modules());scales={}
    for norm_name,names in groups(model):
        layers=[modules[n] for n in names];device,dtype=layers[0].weight.device,layers[0].weight.dtype
        act=torch.stack([stats[n]['absmax'].float() for n in names]).amax(dim=0).to(device=device,dtype=dtype)
        weight=torch.stack([m.weight.abs().amax(dim=0) for m in layers]).amax(dim=0).clamp_min(1e-5)
        scale=(act.pow(alpha)/weight.pow(1-alpha)).clamp_min(1e-5)
        fold_norm(modules[norm_name],scale)
        for m in layers:m.weight.mul_(scale.view(1,-1))
        scales[norm_name]=scale.cpu()
    return scales


@torch.no_grad()
def apply(model, payload, root, activation_bits, require_admission=True):
    if payload['format']!='PI05_TORCH_SMOOTHQUANT_V1' or activation_bits not in [4,8]:
        raise ValueError('Wrong SQ profile/precision')
    if require_admission and payload.get('precision_gates',{}).get(str(activation_bits))!='PASS_RELOAD_FINITE':
        raise ValueError('This activation precision has not passed its actual GPU/reload gate')
    names=target_names(model)
    if set(names)!=set(payload['statistics']):raise ValueError('Profile target coverage mismatch')
    smooth(model,payload['statistics'],payload['alpha'])
    official,_=official_functions(root);modules=dict(model.named_modules());handles=[]
    for name in names:
        module=modules[name]
        expected=payload['statistics'][name]['shape']
        if list(module.weight.shape)!=expected:raise ValueError('Target shape changed')
        weight=module.weight.float().clone()
        official.quantize_weight_per_channel_absmax(weight,n_bits=4)
        module.weight.copy_(weight.to(module.weight.dtype))
        def hook(_module,args,bits=activation_bits):
            value=args[0].detach().clone().contiguous()
            official.quantize_activation_per_token_absmax(value,n_bits=bits)
            return (value,)+tuple(args[1:])
        handles.append(module.register_forward_pre_hook(hook))
    return handles
