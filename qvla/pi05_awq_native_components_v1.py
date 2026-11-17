"""Pinned original AWQ arithmetic plus explicit native-Gemma shape adapters.

This module does not claim full-policy compatibility. Grid search and clipping
are compiled from verified original AST nodes without changing their bodies.
Adaptive normalization folding must leave the residual gate unchanged.
"""
import ast
import gc
import hashlib
from pathlib import Path
import torch
from torch import nn

ORIGINAL_FILES={
    'quantizer.py':'7929ee1f97addde95ce1b9c1d55f027b3b4d13fb1b9e6c850ab8a01736914e6c',
    'auto_scale.py':'b6cf9bc5c7de67a03f56366046d473c1f6633ee6df3f59748b4559d47df858e8',
    'auto_clip.py':'fca53998885661c4930c7a373f52d9dc25e58fb15ad74c87678673ed5209157d',
    'qmodule.py':'310947d8cc49be6036b3408fc38dd3b61e941b18747629910b15415d7deb9658',
}


def original_functions(source,group_size):
    source=Path(source)
    trees={}
    for name,expected in ORIGINAL_FILES.items():
        data=(source/name).read_bytes().replace(b'\r\n',b'\n')
        if hashlib.sha256(data).hexdigest()!=expected:
            raise ValueError('Pinned AWQ source changed: '+name)
        trees[name]=ast.parse(data.decode())
    scope={'torch':torch,'nn':nn,'gc':gc}
    semantic={}
    def load(file,name,kind=ast.FunctionDef):
        node=next(x for x in ast.walk(trees[file]) if isinstance(x,kind) and x.name==name)
        semantic[name]=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source/file),'exec'),scope)
        return scope[name]
    quant=load('quantizer.py','pseudo_quantize_tensor')
    load('auto_scale.py','get_act_scale')
    scope['w_quantize_func']=lambda w:quant(w,n_bit=4,zero_point=True,q_group_size=group_size,inplace=False).detach()
    search=load('auto_scale.py','_search_module_scale')
    clip=load('auto_clip.py','auto_clip_layer')
    scale_ln=load('auto_scale.py','scale_ln_fcs')
    scale_pair=load('auto_scale.py','scale_fc_fc')
    scaled_activation=load('qmodule.py','ScaledActivation',ast.ClassDef)
    return dict(quantize=quant,search=search,clip=clip,scale_layer_norm=scale_ln,
                scale_fc_pair=scale_pair,scaled_activation=scaled_activation,
                original_files=ORIGINAL_FILES,semantic_sha256=semantic)


@torch.no_grad()
def fold_norm_and_fcs(norm,fcs,scales,original):
    fcs=list(fcs)
    if scales.ndim!=1 or not torch.isfinite(scales).all() or not (scales>0).all():
        raise ValueError('Invalid AWQ channel scale')
    if any(fc.in_features!=len(scales) for fc in fcs):
        raise ValueError('AWQ scale differs from the shared input width')
    if isinstance(norm,nn.LayerNorm):
        original['scale_layer_norm'](norm,fcs,scales)
        return
    if type(norm).__name__!='GemmaRMSNorm':
        raise TypeError('Native normalization has no verified AWQ folding')
    dense=getattr(norm,'dense',None)
    if dense is not None and (dense.out_features!=3*len(scales) or dense.bias is None):
        raise ValueError('Unverified adaptive scale/shift/gate layout')
    reference=dense.weight if dense is not None else norm.weight
    scale=scales.to(device=reference.device,dtype=reference.dtype)
    weight=getattr(norm,'weight',None)
    if isinstance(weight,torch.Tensor):
        # The native Gemma gamma is (1+weight), not weight.
        weight.copy_((weight+1)/scale-1)
    if dense is not None:
        width=len(scale)
        gate_weight=dense.weight[2*width:].clone();gate_bias=dense.bias[2*width:].clone()
        dense.weight[:width].div_(scale[:,None])
        dense.bias[:width].copy_((dense.bias[:width]+1)/scale-1)
        dense.weight[width:2*width].div_(scale[:,None])
        dense.bias[width:2*width].div_(scale)
        if not torch.equal(gate_weight,dense.weight[2*width:]) or not torch.equal(gate_bias,dense.bias[2*width:]):
            raise AssertionError('Adaptive residual gate was changed')
    for fc in fcs:
        fc.weight.mul_(scales.to(fc.weight.device,fc.weight.dtype).view(1,-1))
    if not all(torch.isfinite(p).all() for p in norm.parameters()):
        raise ValueError('Nonfinite native norm folding')


@torch.no_grad()
def original_clip_with_output_padding(original,weight,features,group_size):
    """Original clipping is independent per output row; pad only empty rows.

    Native SigLIP fc1 has 4304 outputs, incompatible with the upstream 64/256
    output batches. Padding never changes inputs or any real output-row loss.
    """
    if weight.ndim!=2 or weight.shape[1]%group_size:
        raise ValueError('Registered input groups must divide the actual width')
    if features.numel()//features.shape[-1]<512:
        raise ValueError('Original clipping requires at least 512 token rows')
    rows=weight.shape[0];pad=(-rows)%64
    padded=torch.cat([weight,weight.new_zeros(pad,weight.shape[1])]) if pad else weight
    return original['clip'](padded,features,n_bit=4,q_config=dict(zero_point=True,q_group_size=group_size))[:rows]


@torch.no_grad()
def folded_activation(module,fc,scales,original):
    """Original post-activation divide / next-weight multiply, not a new act."""
    if fc.in_features!=len(scales) or not torch.isfinite(scales).all() or not (scales>0).all():
        raise ValueError('Invalid activation-pair scale')
    fc.weight.mul_(scales.to(fc.weight.device,fc.weight.dtype).view(1,-1))
    result=original['scaled_activation'](module,scales.detach().clone())
    result.requires_grad_(False)
    return result
