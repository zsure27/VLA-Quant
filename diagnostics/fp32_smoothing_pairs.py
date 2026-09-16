"""Promote complete vision norm->linear pairs, return to source output dtype.

Numerical diagnostic only: this increases parameter memory and is not PEFT.
"""
import torch
from torch import nn

def promote_pairs(model, groups):
    modules=dict(model.named_modules())
    handles=[]; promoted=[]
    for norm_name,names,_ in groups:
        if not norm_name.startswith("vision_backbone.") or len(names)!=1:
            raise ValueError("FP32 numerical pairs require complete single-target vision groups")
        norm=modules[norm_name]; fc=modules[names[0]]
        if not isinstance(norm,nn.LayerNorm) or not isinstance(fc,nn.Linear):
            raise TypeError("FP32 pairs require LayerNorm and Linear")
        dtype=fc.weight.dtype
        norm.float(); fc.float()
        def input_float(_module,args): return (args[0].float(),)+args[1:]
        def output_cast(_module,_args,output,dtype=dtype): return output.to(dtype)
        handles.append(norm.register_forward_pre_hook(input_float))
        handles.append(fc.register_forward_hook(output_cast))
        promoted.append({"norm":norm_name,"linear":names[0],"output_dtype":str(dtype),
                         "fp32_parameter_elements":sum(p.numel() for m in (norm,fc) for p in m.parameters())})
    return handles,promoted
