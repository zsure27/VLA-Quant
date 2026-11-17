"""CPU-only inference-path and adaptive-normalization preparation, not AWQ admission.

The algebraic test below does not run AWQ scale search or clipping. No full model
is allocated. Original source, calibration and current GPU programs stay read-only.
"""
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import torch
from transformers import GemmaConfig
from transformers.models.gemma import modeling_gemma
torch.set_num_threads(1)
torch.manual_seed(7)
assert not torch.cuda.is_available()
a.output.mkdir(parents=True,exist_ok=False)
src=a.root/'sources/openpi/src/openpi/models_pytorch'
policy=src/'pi0_pytorch.py';gemma=src/'gemma_pytorch.py'
tree=ast.parse(policy.read_text())
calls=[]
for fn in ast.walk(tree):
    if not isinstance(fn,ast.FunctionDef) or fn.name not in ['sample_actions','denoise_step']:continue
    for node in ast.walk(fn):
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute) or node.func.attr!='forward':continue
        for kw in node.keywords:
            if kw.arg=='inputs_embeds':calls.append(dict(function=fn.name,expression=ast.unparse(kw.value)))
assert {x['expression'] for x in calls}=={'[prefix_embs, None]','[None, suffix_embs]'},calls
text=gemma.read_text();assert 'elif inputs_embeds[0] is None:' in text
assert 'self.gemma_expert.model.forward(' in text and 'self.paligemma.language_model.forward(' in text
checks=[]
for adaptive in [False,True]:
    cfg=GemmaConfig(hidden_size=64,intermediate_size=128,num_hidden_layers=1,num_attention_heads=8,
                    num_key_value_heads=1,head_dim=8,vocab_size=128,use_adarms=adaptive,
                    adarms_cond_dim=64 if adaptive else None,hidden_activation='gelu_pytorch_tanh')
    cfg._attn_implementation='eager'
    layer=modeling_gemma.GemmaDecoderLayer(cfg,layer_idx=0).eval()
    norm=layer.input_layernorm
    original={k:v.detach().clone() for k,v in norm.state_dict().items()}
    x=torch.randn(1,6,64);cond=torch.randn(1,64) if adaptive else None
    with torch.no_grad():y,gate=norm(x,cond=cond)
    scales=torch.linspace(.5,2.,64);weight=torch.randn(9,64)
    expected=torch.nn.functional.linear(y,weight)
    actual=torch.nn.functional.linear(y/scales,weight*scales)
    error=float((actual-expected).abs().max());assert error<1e-4 and torch.isfinite(actual).all()
    assert all(torch.equal(v,norm.state_dict()[k]) for k,v in original.items())
    # The residual gate is deliberately outside the scaling transformation.
    if gate is not None:assert torch.isfinite(gate).all()
    state=dict(norm=norm.state_dict(),scales=scales)
    path=a.output/f'tiny-adaptive-{adaptive}-preparation.pt';torch.save(state,path)
    loaded=torch.load(path,map_location='cpu',weights_only=True)
    assert torch.equal(scales,loaded['scales'])
    checks.append(dict(adaptive=adaptive,shape=list(y.shape),fp32_compensation_max_abs=error,
                       norm_parameters_unchanged=True,residual_gate_not_transformed=True,
                       small_state_reload_bitwise=True))
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
out=dict(gate='PASS_PI05_INFERENCE_BRANCH_AND_ADARMS_SCALING_COMPONENT_CPU',
         source_sha256={policy.name:sha(policy),gemma.name:sha(gemma)},inference_calls=calls,
         inference_scope='Separate prefix VLM prefill and cached expert denoising; joint compute_layer_complete is the training path',
         native_norm_source_sha256=hashlib.sha256(inspect.getsource(modeling_gemma.GemmaRMSNorm).encode()).hexdigest(),
         checks=checks,full_model_loaded=False,gpu_model_admitted=False,full_model_adapter_complete=False,
         awq_scale_search_run=False,awq_clipping_run=False,
         remaining=['Exact native block kwargs/KV/position/condition capture',
                    'Original AWQ 20-grid search and clipping with registered 128/16 groups',
                    'Full unquantized action equivalence and quantized finite/save/reload gates',
                    'Same-input/noise four-suite pilot before formal trajectories'])
(a.output/'result.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
