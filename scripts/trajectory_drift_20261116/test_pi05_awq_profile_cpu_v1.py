"""63 tiny native blocks exercise loader contracts; not a real VLA policy."""
import argparse,os,json,copy
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--blocks',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import torch
from torch import nn
from transformers import GemmaConfig,SiglipVisionConfig
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer
from qvla.pi05_awq_profile_v1 import apply_profile,BACKEND_ID,target_names
from qvla.infra.common import digest
torch.set_num_threads(1);torch.manual_seed(7);assert not torch.cuda.is_available();a.output.mkdir(exist_ok=False,parents=True)
def gemma(adaptive):
 c=GemmaConfig(hidden_size=128,intermediate_size=256,num_hidden_layers=18,num_attention_heads=8,num_key_value_heads=1,head_dim=16,vocab_size=128,use_adarms=adaptive,adarms_cond_dim=128 if adaptive else None,hidden_activation='gelu_pytorch_tanh');c._attn_implementation='eager'
 x=GemmaDecoderLayer(c,layer_idx=0).eval();x.input_layernorm.to(torch.float32);x.post_attention_layernorm.to(torch.float32);return x
v=SiglipVisionConfig(hidden_size=128,intermediate_size=256,num_hidden_layers=27,num_attention_heads=4);v._attn_implementation='eager'
model=nn.Module();parts={'prefix':gemma(False),'expert':gemma(True),'vision':SiglipEncoderLayer(v).eval()};entries=[]
for kind,path,count in [('prefix','paligemma_with_expert.paligemma.model.language_model.layers',18),('expert','paligemma_with_expert.gemma_expert.model.layers',18),('vision','paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers',27)]:
 node=model
 for name in path.split('.')[:-1]:
  if not hasattr(node,name):setattr(node,name,nn.Module())
  node=getattr(node,name)
 setattr(node,'layers',nn.ModuleList([copy.deepcopy(parts[kind]) for _ in range(count)]))
 for i in range(count):entries.append(dict(name=path+'.'+str(i),file=kind+'.pt',sha256=digest(a.blocks/(kind+'.pt'))))
model.eval().requires_grad_(False);assert len(target_names(model))==414
manifest=dict(backend=BACKEND_ID,method='awq_w4a16_native_blocks_v1',entries=entries,full_policy_admitted=False)
checks=[]
for name,wrong in [('pending_profile',manifest),('missing_block',dict(manifest,entries=entries[:-1]))]:
 try:apply_profile(model,wrong,a.blocks,a.source)
 except ValueError:checks.append(name+'_rejected')
 else:raise AssertionError(name+' accepted')
one=copy.deepcopy(model);two=copy.deepcopy(model)
apply_profile(one,manifest,a.blocks,a.source,allow_pending=True);apply_profile(two,json.loads(json.dumps(manifest)),a.blocks,a.source,allow_pending=True)
assert all(torch.equal(v,two.state_dict()[n]) for n,v in one.state_dict().items())
assert all(not x.requires_grad for x in one.parameters()) and all(torch.isfinite(x).all() for x in one.parameters())
bad=copy.deepcopy(manifest);bad['entries'][0]['sha256']='0'*64
try:apply_profile(copy.deepcopy(model),bad,a.blocks,a.source,allow_pending=True)
except ValueError:checks.append('changed_artifact_digest_rejected')
else:raise AssertionError('Changed artifact accepted')
out=dict(gate='PASS_AWQ_63_TINY_NATIVE_BLOCK_PROFILE_LOAD_CONTRACT_CPU',checks=checks,blocks=63,projection_names=414,repeat_load_state_bitwise=True,frozen_finite=True,real_vla_policy=False,full_policy_gpu_admitted=False,formal_episodes=0)
(a.output/'result.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
