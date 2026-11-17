"""Adapter for original AWQ full-module scoring on native policy call contexts.

No quantization algorithm is changed. Attention keeps each original mask,
position and layer-local KV snapshot, including expert denoising contexts.
Only one block is retained; serialized activation caches are not built.
"""
import torch
from torch import nn
from qvla.pi05_block_replay_v2 import replay_call


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel()*value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple,list)):
        return sum(map(tensor_bytes,value))
    return 0


class ReplayBatch(nn.Module):
    """Concatenate independent call losses without mixing their attention masks."""
    def __init__(self, module, captures, max_cpu_bytes=2*1024**3):
        super().__init__()
        if module.training or not captures:
            raise ValueError('An eval module and nonempty calibration calls are required')
        if any(c.get('schema_version')!=1 for c in captures):
            raise ValueError('Unknown native capture schema')
        if tensor_bytes(captures)>max_cpu_bytes:
            raise ValueError('Current-block calibration capture exceeds memory budget')
        widths={c['hidden'].shape[-1] for c in captures}
        if len(widths)!=1:
            raise ValueError('One AWQ scoring group must have one input width')
        self.block=module
        self.captures=list(captures)
        self.width=widths.pop()
        self.shapes=[tuple(c['hidden'].shape) for c in captures]
        self.rows=[c['hidden'].numel()//self.width for c in captures]
        self.eval()

    def inputs(self):
        return torch.cat([c['hidden'].reshape(-1,self.width) for c in self.captures])

    def state_dict(self,*args,**kwargs):
        state=super().state_dict(*args,**kwargs)
        # Upstream snapshots CUDA weights using .cpu(). On a CPU fixture that
        # would alias storage; make the same independent snapshot explicitly.
        return {k:v.clone() if v.device.type=='cpu' else v for k,v in state.items()}

    def forward(self, hidden):
        if hidden.ndim!=2 or tuple(hidden.shape)!=(sum(self.rows),self.width):
            raise ValueError('Scoring inputs differ from captured token layout')
        results=[]
        for part,shape,capture in zip(hidden.split(self.rows),self.shapes,self.captures):
            value=replay_call(self.block,dict(capture,hidden=part.reshape(shape)))
            if isinstance(value,tuple):
                value=value[0]
            if not isinstance(value,torch.Tensor) or not torch.isfinite(value).all():
                raise ValueError('Nonfinite or unsupported native replay output')
            results.append(value.reshape(-1,value.shape[-1]))
        return torch.cat(results)


def native_scale_groups(block):
    """Known native structure, checked before applying any shared scale."""
    kind=type(block).__name__
    if kind=='GemmaDecoderLayer':
        groups=[dict(previous='input_layernorm',targets=['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj'],score='self_attn',fold='norm'),
                dict(previous='post_attention_layernorm',targets=['mlp.gate_proj','mlp.up_proj'],score='mlp',fold='norm'),
                dict(previous='mlp.up_proj',targets=['mlp.down_proj'],score='mlp.down_proj',fold='linear_pair')]
        if block.self_attn.v_proj.weight.shape==block.self_attn.o_proj.weight.shape:
            groups.insert(1,dict(previous='self_attn.v_proj',targets=['self_attn.o_proj'],score='self_attn.o_proj',fold='linear_pair'))
    elif kind=='SiglipEncoderLayer':
        groups=[dict(previous='layer_norm1',targets=['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj'],score='self_attn',fold='norm'),
                dict(previous='self_attn.v_proj',targets=['self_attn.out_proj'],score='self_attn.out_proj',fold='linear_pair'),
                dict(previous='layer_norm2',targets=['mlp.fc1'],score='mlp',fold='norm'),
                dict(previous='mlp.activation_fn',targets=['mlp.fc2'],score='mlp.fc2',fold='activation_pair')]
    else:
        raise TypeError('Unverified native block type: '+kind)
    for group in groups:
        targets=[block.get_submodule(n) for n in group['targets']]
        if any(not isinstance(m,nn.Linear) for m in targets):
            raise ValueError('AWQ target is not the native linear projection')
        if len({m.in_features for m in targets})!=1:
            raise ValueError('Shared AWQ input widths differ')
        block.get_submodule(group['previous']);block.get_submodule(group['score'])
    return groups
