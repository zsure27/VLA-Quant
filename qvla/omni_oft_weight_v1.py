"""Original OmniQuant LWC weight-only wrappers, independent OFT admission."""
import importlib.util
from pathlib import Path
import torch
import torch.nn.functional as F
from qvla.baseline_contract import check_scope, CONTRACT_ID

ROOT=Path('/root/autodl-tmp/trajectory-drift-20261107')
# Shared lossless codec only; GPTQ's algorithm is not called by this adapter.
spec=importlib.util.spec_from_file_location('lossless_uint4_codec',ROOT/'method-code/20261110-gptq-v3/gptq_oft_v1.py')
codec=importlib.util.module_from_spec(spec);spec.loader.exec_module(codec)
pack=codec.pack
unpack=codec.unpack
weight_sha=codec.weight_sha

class QuantConvLWC(torch.nn.Module):
    def __init__(self, module, quantizer_class):
        super().__init__();self.register_buffer('weight',module.weight.detach().clone());self.register_buffer('bias',module.bias.detach().clone() if module.bias is not None else None)
        self.stride=module.stride;self.padding=module.padding;self.dilation=module.dilation;self.groups=module.groups;self.use_weight_quant=False
        # Original asymmetric grouped LWC requires divisible widths. Patch
        # kernels use original per-row LWC, explicitly recorded in each entry.
        shape=self.weight.flatten(1).shape
        self.weight_quantizer=quantizer_class(n_bits=4,per_channel_axes=[0],symmetric=False,dynamic_method='per_channel',group_size=None,lwc=True,shape=shape)
    def forward(self,x):
        w=self.weight_quantizer(self.weight.flatten(1)).reshape_as(self.weight) if self.use_weight_quant else self.weight
        return F.conv2d(x,w,self.bias,self.stride,self.padding,self.dilation,self.groups)
    def set_quant_state(self,weight_quant=False,act_quant=False):
        assert not act_quant;self.use_weight_quant=weight_quant

@torch.no_grad()
def export_weight(module):
    q=module.weight_quantizer;w=module.weight.flatten(1);value=q(w).reshape_as(module.weight).to(torch.bfloat16)
    rows,cols=w.shape;group=q.group_size or cols;n=(cols+group-1)//group
    scales=q.scale.reshape(rows,n);zeros=q.round_zero_point.reshape(rows,n)
    entry=pack(value,[scales[:,i] for i in range(n)],[zeros[:,i] for i in range(n)],group)
    entry['lwc_upbound_factor']=q.upbound_factor.detach().cpu();entry['lwc_lowbound_factor']=q.lowbound_factor.detach().cpu()
    return value,entry

def load_profiles(paths,method):
    assert method=='omniquant' and len(paths)==1
    payload=torch.load(Path(paths[0]),map_location='cpu',weights_only=True)
    assert payload['format']=='OFT_ORIGINAL_OMNI_LWC_PACK_V1' and payload['contract_id']==CONTRACT_ID
    assert payload['gate']=='PASS_FULL_422_OMNI_LWC_PACK_AND_RELOAD';check_scope(payload['entries'])
    return payload['entries'],payload

@torch.no_grad()
def apply_quantization(model,entries,metadata,method,official,weight_bits,activation_bits):
    assert method=='omniquant' and weight_bits==4 and activation_bits==16;check_scope(entries);modules=dict(model.named_modules())
    for name,entry in entries.items():
        m=modules[name];w=unpack(entry,m.weight.device,m.weight.dtype);assert torch.isfinite(w).all() and weight_sha(w)==entry['weight_sha256'];m.weight.copy_(w)
    return []

validate_calibration_actions=codec.validate_calibration_actions
