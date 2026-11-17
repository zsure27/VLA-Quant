"""Verify variable native contexts and unchanged original AWQ scoring on CPU."""
import argparse,json,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import torch
from transformers import GemmaConfig,SiglipVisionConfig
from transformers.models.gemma import modeling_gemma as native
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer
from qvla.pi05_block_replay_v2 import capture_call,replay_call,LayerKVCache
from qvla.pi05_awq_replay_batch_v1 import ReplayBatch,native_scale_groups,tensor_bytes
from qvla.pi05_awq_native_components_v2 import original_functions
torch.set_num_threads(1);torch.manual_seed(7);assert not torch.cuda.is_available()
a.output.mkdir(parents=True,exist_ok=False);checks=[]
for adaptive in [False,True]:
 cfg=GemmaConfig(hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=8,num_key_value_heads=1,head_dim=8,vocab_size=128,use_adarms=adaptive,adarms_cond_dim=64 if adaptive else None,hidden_activation='gelu_pytorch_tanh');cfg._attn_implementation='eager'
 layer=native.GemmaDecoderLayer(cfg,layer_idx=1).eval();rope=native.GemmaRotaryEmbedding(config=cfg);calls=[]
 for cached,length in [(False,5),(True,3)]:
  prefix=7 if cached else 0;x=torch.randn(1,length,64);positions=torch.arange(prefix,prefix+length)[None]
  pair=(torch.randn(1,1,prefix,8),torch.randn(1,1,prefix,8)) if cached else None
  kwargs=dict(position_ids=positions,attention_mask=torch.zeros(1,1,length,length+prefix),position_embeddings=rope(x,positions),past_key_value=LayerKVCache(1,pair),use_cache=not cached,cache_position=positions[0],adarms_cond=torch.randn(1,64) if adaptive else None)
  calls.append(capture_call((x,),kwargs,1))
 batch=ReplayBatch(layer,calls);expected=torch.cat([replay_call(layer,c)[0].reshape(-1,64) for c in calls]);actual=batch(batch.inputs())
 assert torch.equal(expected,actual) and torch.equal(actual,batch(batch.inputs()))
 groups=native_scale_groups(layer);assert len(groups)==3 and not any(g['previous']=='self_attn.v_proj' for g in groups)
 try:ReplayBatch(layer,calls,max_cpu_bytes=tensor_bytes(calls)-1)
 except ValueError:pass
 else:raise AssertionError('Memory budget not enforced')
 try:batch(batch.inputs()[:-1])
 except ValueError:pass
 else:raise AssertionError('Changed token layout accepted')
 checks.append(dict(adaptive=adaptive,variable_token_contexts_bitwise=True,fresh_KV_per_call=True,gqa_value_output_scale_skipped=True,memory_and_layout_rejected=True))
cfg=SiglipVisionConfig(hidden_size=64,intermediate_size=128,num_hidden_layers=1,num_attention_heads=4);cfg._attn_implementation='eager';vision=SiglipEncoderLayer(cfg).eval()
calls=[capture_call((torch.randn(1,n,64),),dict(attention_mask=None),0) for n in [5,8]]
batch=ReplayBatch(vision,calls);expected=torch.cat([replay_call(vision,c)[0].reshape(-1,64) for c in calls]);assert torch.equal(batch(batch.inputs()),expected)
assert len(native_scale_groups(vision))==4
# Original 20-grid search scores the whole MLP, preserving activation and output
# projection rather than substituting a single-linear proxy for that loss.
mlp=vision.mlp;calls=[capture_call((torch.randn(1,n,64),),{},0) for n in [32,33]];batch=ReplayBatch(mlp,calls)
before={k:v.detach().clone() for k,v in mlp.state_dict().items()};original=original_functions(a.source,16)
with torch.no_grad():
 scale=original['search'](batch,[mlp.fc1],batch.inputs())
assert scale.shape==(64,) and torch.isfinite(scale).all() and (scale>0).all()
assert all(torch.equal(v,mlp.state_dict()[k]) for k,v in before.items())
checks.append(dict(component='siglip_full_block_and_mlp',variable_token_contexts_bitwise=True,whole_mlp_original_20_grid_search=True,parameters_restored_bitwise=True))
out=dict(gate='PASS_NATIVE_AWQ_VARIABLE_CONTEXT_BATCH_AND_FULL_MODULE_SEARCH_CPU',checks=checks,full_model_loaded=False,full_awq_profile_built=False,gpu_method_admitted=False)
(a.output/'result.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
