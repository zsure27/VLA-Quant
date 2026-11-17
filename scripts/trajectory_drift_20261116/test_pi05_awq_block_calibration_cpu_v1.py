"""Tiny real native block AWQ integration; does not admit a full policy."""
import argparse,os,json,copy,inspect
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import torch
from transformers import GemmaConfig,SiglipVisionConfig
from transformers.models.gemma import modeling_gemma as native
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer
from qvla.pi05_block_replay_v2 import capture_call,replay_call
from qvla.pi05_awq_block_calibration_v1 import calibrate_block,load_block,native_scale_groups
torch.set_num_threads(1);torch.manual_seed(7);assert not torch.cuda.is_available()
a.output.mkdir(exist_ok=False,parents=True);checks=[]
for kind in ['prefix','expert','vision']:
 if kind=='vision':
  cfg=SiglipVisionConfig(hidden_size=128,intermediate_size=256,num_hidden_layers=1,num_attention_heads=4);cfg._attn_implementation='eager';block=SiglipEncoderLayer(cfg).eval()
 else:
  adaptive=kind=='expert';cfg=GemmaConfig(hidden_size=128,intermediate_size=256,num_hidden_layers=1,num_attention_heads=8,num_key_value_heads=1,head_dim=16,vocab_size=128,use_adarms=adaptive,adarms_cond_dim=128 if adaptive else None,hidden_activation='gelu_pytorch_tanh');cfg._attn_implementation='eager';block=native.GemmaDecoderLayer(cfg,layer_idx=0).eval();rope=native.GemmaRotaryEmbedding(config=cfg)
 block.requires_grad_(False);before={n:v.detach().clone() for n,v in block.state_dict().items()};groups=native_scale_groups(block)
 features={n:[] for n,m in block.named_modules() if isinstance(m,torch.nn.Linear) and not n.endswith('.dense')};calls={g['score']:[] for g in groups};hooks=[];full=[]
 def pre_score(name,module,args,kwargs):
  bound=dict(inspect.signature(module.forward).bind(*args,**kwargs).arguments)
  # Linear uses input; native attention/MLP uses hidden_states.
  hidden=bound.pop('hidden_states') if 'hidden_states' in bound else bound.pop('input') if 'input' in bound else bound.pop('x')
  for key,param in inspect.signature(module.forward).parameters.items():
   if param.kind==inspect.Parameter.VAR_KEYWORD and key in bound:bound.update(bound.pop(key))
  calls[name].append(capture_call((hidden,),bound,0))
 for name in calls:
  hooks.append(block.get_submodule(name).register_forward_pre_hook(lambda m,x,k,name=name:pre_score(name,m,x,k),with_kwargs=True))
 for name in features:
  hooks.append(block.get_submodule(name).register_forward_pre_hook(lambda m,x,name=name:features[name].append(x[0].detach().cpu().reshape(-1,x[0].shape[-1]).clone())))
 try:
  for j in range(2):
   x=torch.randn(1,256,128)
   if kind=='vision':kw=dict(attention_mask=None)
   else:
    pos=torch.arange(256)[None];kw=dict(position_ids=pos,attention_mask=torch.zeros(1,1,256,256),position_embeddings=rope(x,pos),past_key_value=None,use_cache=False,cache_position=pos[0],adarms_cond=torch.randn(1,128) if kind=='expert' else None)
   full.append(capture_call((x,),kw,0))
   replay_call(block,full[-1])
 finally:
  for h in hooks:h.remove()
 features={n:torch.cat(x) for n,x in features.items()}
 artifact=calibrate_block(block,calls,features,a.source)
 assert all(torch.equal(v,block.state_dict()[n]) for n,v in before.items())
 path=a.output/(kind+'.pt');torch.save(artifact,path);restored=torch.load(path,weights_only=True)
 one=copy.deepcopy(block);two=copy.deepcopy(block);load_block(one,artifact,a.source);load_block(two,restored,a.source)
 for call in full:
  unwrap=lambda v:v[0] if isinstance(v,tuple) else v
  out=unwrap(replay_call(one,call));other=unwrap(replay_call(two,call))
  assert torch.isfinite(out).all() and torch.equal(out,other)
 assert any(not torch.equal(v,one.state_dict()[n]) for n,v in before.items())
 bad=dict(restored,weights=dict(restored['weights']));bad['weights'].pop(next(iter(bad['weights'])))
 try:load_block(copy.deepcopy(block),bad,a.source)
 except ValueError:pass
 else:raise AssertionError('Missing projection accepted')
 checks.append(dict(component=kind,packed_projections=len(artifact['weights']),teacher_restored_bitwise=True,serialized_reload_output_bitwise=True,finite=True,missing_projection_rejected=True))
out=dict(gate='PASS_NATIVE_BLOCK_AWQ_SEARCH_FOLD_CLIP_PACK_RELOAD_CPU',checks=checks,full_policy_loaded=False,gpu_method_admitted=False,scope='Actual tiny prefix/expert/vision blocks; full-policy capture/profile/admission still pending')
(a.output/'result.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
