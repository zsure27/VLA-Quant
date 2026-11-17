"""Pinned original search/clip and native folding tests; no full policy loaded."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--module',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import torch
from torch import nn
from transformers import GemmaConfig
from transformers.models.gemma import modeling_gemma as native
from qvla.pi05_packed_weight_v1 import pack,unpack
assert not torch.cuda.is_available();torch.set_num_threads(1);torch.manual_seed(7)
spec=importlib.util.spec_from_file_location('awq_native_components',a.module)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
source=a.root/'sources/quant-source-sparse-v2/llm-awq/awq/quantize'
original=m.original_functions(source,16)
a.output.mkdir(parents=True,exist_ok=False)
checks=[]
for adaptive in [False,True]:
    cfg=GemmaConfig(hidden_size=64,intermediate_size=128,num_hidden_layers=1,num_attention_heads=8,
                   num_key_value_heads=1,head_dim=8,vocab_size=128,use_adarms=adaptive,
                   adarms_cond_dim=64 if adaptive else None)
    layer=native.GemmaDecoderLayer(cfg,layer_idx=0).eval().requires_grad_(False)
    norm=layer.input_layernorm;fcs=[layer.self_attn.q_proj,layer.self_attn.k_proj,layer.self_attn.v_proj]
    x=torch.randn(2,17,64);cond=torch.randn(2,64) if adaptive else None
    scales=torch.linspace(.5,2.,64)
    with torch.no_grad():
        y,gate=norm(x,cond=cond);expected=[fc(y).clone() for fc in fcs]
        m.fold_norm_and_fcs(norm,fcs,scales,original)
        y2,gate2=norm(x,cond=cond);actual=[fc(y2) for fc in fcs]
    error=max(float((u-v).abs().max()) for u,v in zip(expected,actual))
    assert error<1e-5 and (gate is None or torch.equal(gate,gate2))
    checks.append(dict(kind='native_adaptive_norm' if adaptive else 'native_one_plus_gamma_norm',max_abs=error,residual_gate_unchanged=True))
    inp=torch.randn(2,17,64)
    with torch.no_grad():
        before=layer.mlp(inp).clone()
        original['scale_fc_pair'](layer.mlp.up_proj,layer.mlp.down_proj,torch.linspace(.5,2.,128))
        after=layer.mlp(inp)
    error=float((before-after).abs().max());assert error<1e-5
    checks.append(dict(kind='native_gated_up_down_pair',max_abs=error))
norm=nn.LayerNorm(64).eval().requires_grad_(False);fc=nn.Linear(64,64).eval().requires_grad_(False)
x=torch.randn(2,17,64)
with torch.no_grad():
    before=fc(norm(x)).clone();m.fold_norm_and_fcs(norm,[fc],torch.linspace(.5,2.,64),original);after=fc(norm(x))
assert (before-after).abs().max()<1e-5
activation=nn.GELU();fc=nn.Linear(128,64).eval().requires_grad_(False);x=torch.randn(2,17,128)
with torch.no_grad():
    before=fc(activation(x)).clone();wrapped=m.folded_activation(activation,fc,torch.linspace(.5,2.,128),original);after=fc(wrapped(x))
assert (before-after).abs().max()<1e-5 and not any(p.requires_grad for p in wrapped.parameters())
checks.append(dict(kind='original_layer_norm_and_scaled_activation',passed=True))

class CPUOnlyProbe(nn.Module):
    def __init__(self,module):super().__init__();self.module=module
    def forward(self,x):return self.module(x)
    def state_dict(self,*args,**kwargs):
        # Upstream runs on CUDA: .cpu() copies. On CPU provide the same independent
        # storage so its unchanged restore algorithm does not alias live weights.
        return {k:v.detach().clone() for k,v in super().state_dict(*args,**kwargs).items()}

layer=native.GemmaDecoderLayer(cfg,layer_idx=0).eval().requires_grad_(False)
probe=CPUOnlyProbe(layer.mlp);weights={k:v.clone() for k,v in layer.mlp.state_dict().items()}
with torch.no_grad():scales=original['search'](probe,[layer.mlp.gate_proj,layer.mlp.up_proj],torch.randn(512,64))
assert scales.shape==(64,) and torch.isfinite(scales).all() and (scales>0).all()
assert all(torch.equal(v,layer.mlp.state_dict()[k]) for k,v in weights.items())
checks.append(dict(kind='unchanged_original_20_grid_search_cpu_copy_semantics',parameters_restored_bitwise=True))
weight=torch.randn(4304,16);features=torch.randn(512,16);before=weight.clone()
clip=m.original_clip_with_output_padding(original,weight,features,16)
first=original['clip'](weight[:64],features,4,dict(zero_point=True,q_group_size=16))
last=m.original_clip_with_output_padding(original,weight[-16:],features,16)
assert torch.equal(clip[:64],first) and torch.equal(clip[-16:],last) and torch.equal(weight,before)
checks.append(dict(kind='original_clip_4304_output_row_padding',first_and_tail_rows_bitwise=True,weights_unchanged=True))
for width,group in [(128,128),(4304,16)]:
    for dtype in [torch.float32,torch.bfloat16]:
        weight=torch.randn(64,width).to(dtype)
        q,sc,ze=original['quantize'](weight,4,True,group,False,True)
        packed=pack(q,list(sc.unbind(1)),list(ze.unbind(1)),group)
        assert torch.equal(q,unpack(packed,'cpu',dtype))
        path=a.output/f'packing-width{width}-{str(dtype).split(".")[-1]}.pt';torch.save(packed,path)
        assert torch.equal(q,unpack(torch.load(path,weights_only=True),'cpu',dtype))
checks.append(dict(kind='original_awq_4bit_group128_and16_packing_float32_bfloat16',lossless=True))
result=dict(gate='PASS_PINNED_AWQ_NATIVE_FOLD_SEARCH_CLIP_PACK_CPU',checks=checks,
            original_files=original['original_files'],semantic_sha256=original['semantic_sha256'],
            full_model_loaded=False,gpu_model_admitted=False,full_awq_profile_not_built=True,
            cpu_search_state_copy_only=True,source_algorithms_unchanged=True)
(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
