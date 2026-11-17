"""Prove cache-layout-only change keeps native captures and AWQ outputs exact."""
import argparse,os,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import torch,numpy as np
from transformers import GemmaConfig,SiglipVisionConfig
from transformers.models.gemma import modeling_gemma as native
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer
from qvla.pi05_awq_capture_block_v1 import collect as old_collect,BACKEND_ID
from qvla.pi05_awq_capture_block_v2 import collect as new_collect
from qvla.pi05_awq_block_calibration_v1 import calibrate_block,unique_tensor_bytes
from qvla.pi05_block_replay_v2 import LayerKVCache
torch.set_num_threads(1);torch.manual_seed(7);assert not torch.cuda.is_available();a.output.mkdir(exist_ok=False,parents=True)
checks=[]
def equal_tree(one,two):
 if isinstance(one,torch.Tensor):return isinstance(two,torch.Tensor) and torch.equal(one,two)
 if isinstance(one,dict):return one.keys()==two.keys() and all(equal_tree(v,two[k]) for k,v in one.items())
 if isinstance(one,(tuple,list)):return len(one)==len(two) and all(equal_tree(x,y) for x,y in zip(one,two))
 return one==two
for kind in ['prefix','expert','vision']:
 if kind=='vision':
  cfg=SiglipVisionConfig(hidden_size=128,intermediate_size=256,num_hidden_layers=1,num_attention_heads=4);cfg._attn_implementation='eager';block=SiglipEncoderLayer(cfg).eval().to(torch.bfloat16)
 else:
  cfg=GemmaConfig(hidden_size=128,intermediate_size=256,num_hidden_layers=1,num_attention_heads=8,num_key_value_heads=1,head_dim=16,vocab_size=128,use_adarms=kind=='expert',adarms_cond_dim=128 if kind=='expert' else None,hidden_activation='gelu_pytorch_tanh');cfg._attn_implementation='eager';block=native.GemmaDecoderLayer(cfg,layer_idx=0).eval().to(torch.bfloat16);block.input_layernorm.to(torch.float32);block.post_attention_layernorm.to(torch.float32);rope=native.GemmaRotaryEmbedding(config=cfg)
 block.requires_grad_(False)
 class Fixture:
  def __init__(self):self._model=block
  def infer(self,obs,noise):
   x=torch.as_tensor(obs['observation/image']).to(torch.bfloat16)
   if kind=='vision':kw=dict(attention_mask=None)
   else:
    prefix=7 if kind=='expert' else 0;pos=torch.arange(prefix,prefix+16)[None];pair=(torch.zeros(1,1,prefix,16,dtype=torch.bfloat16),torch.zeros(1,1,prefix,16,dtype=torch.bfloat16)) if prefix else None
    kw=dict(position_ids=pos,attention_mask=torch.zeros(1,1,16,16+prefix),position_embeddings=rope(x,pos),past_key_value=LayerKVCache(0,pair),use_cache=not bool(prefix),cache_position=pos[0],adarms_cond=torch.as_tensor(noise[0]).repeat(4)[None] if kind=='expert' else None)
   with torch.no_grad():value=self._model(x,**kw)[0]
   return {'actions':value[0,:10,:7].float().numpy().copy()}
 policy=Fixture();rows=[];rng=np.random.default_rng(7)
 for s in range(4):
  folder=a.output/kind/f'fixture-suite{s}';folder.mkdir(parents=True)
  (folder/'result.json').write_text(json.dumps(dict(role='calibration',task_id=0,reset_index=0,condition='bf16',backend=BACKEND_ID,suite=f'fixture-suite{s}',gate='PASS_PI05_TORCH_CPU_FIXTURE',real_vla_episode=False)))
  for i in range(8):
   image=rng.standard_normal((1,16,128)).astype(np.float32);noise=rng.standard_normal((10,32)).astype(np.float32);out=policy.infer({'observation/image':image},noise)['actions'];file=folder/f'query-{i}.npz'
   np.savez_compressed(file,image=image,wrist_image=np.zeros((2,2,3),np.uint8),state=np.zeros(8,np.float32),prompt=np.array('CPU fixture only'),policy_noise=noise,raw_policy_chunk=out)
   rows.append(dict(path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest(),suite=f'fixture-suite{s}',role='calibration',reset_index=0))
 old_calls,old_features=old_collect(policy,block,rows);new_calls,new_features=new_collect(policy,block,rows)
 assert equal_tree(old_calls,new_calls) and equal_tree(old_features,new_features)
 before=unique_tensor_bytes([old_calls,old_features]);after=unique_tensor_bytes([new_calls,new_features]);assert after<before
 new_collect(policy,block,rows,max_cpu_bytes=after+1)
 try:old_collect(policy,block,rows,max_cpu_bytes=after+1)
 except ValueError:pass
 else:raise AssertionError('Memory reduction was not necessary/effective in fixture')
 original=calibrate_block(block,old_calls,old_features,a.source)
 revised=calibrate_block(block,new_calls,new_features,a.source)
 assert equal_tree(original,revised)
 assert all(not m._forward_pre_hooks for m in block.modules())
 checks.append(dict(component=kind,old_cache_bytes=before,new_cache_bytes=after,all_captured_values_bitwise=True,all_awq_artifact_tensors_bitwise=True,unchanged_stricter_budget_passed=True))
out=dict(gate='PASS_AWQ_CACHE_LAYOUT_VALUES_AND_QUANT_ARTIFACTS_BITWISE_CPU',checks=checks,production_cache_budget_bytes=2*1024**3,calibration_queries_unchanged=32,original_algorithm_unchanged=True,real_vla_policy=False,formal_episodes=0)
(a.output/'result.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
