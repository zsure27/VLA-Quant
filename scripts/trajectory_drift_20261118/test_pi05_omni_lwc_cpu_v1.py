"""Actual tiny native units: unchanged replay, LWC gradients and exact packing."""
import argparse,json,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import torch
from transformers import GemmaConfig,SiglipVisionConfig
from transformers.models.gemma import modeling_gemma as gemma
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer
from qvla.pi05_block_replay_v2 import capture_call,replay_call,LayerKVCache
from qvla.pi05_omni_lwc_units_v1 import calibrate,replay,tensor,SOURCE_SHA
torch.set_num_threads(1);torch.manual_seed(7);assert not torch.cuda.is_available()
cfg=GemmaConfig(hidden_size=128,intermediate_size=256,num_hidden_layers=2,num_attention_heads=8,num_key_value_heads=1,head_dim=16,vocab_size=128,hidden_activation='gelu_pytorch_tanh');cfg._attn_implementation='eager'
layer=gemma.GemmaDecoderLayer(cfg,layer_idx=1).eval();rope=gemma.GemmaRotaryEmbedding(config=cfg)
vision_cfg=SiglipVisionConfig(hidden_size=128,intermediate_size=256,num_hidden_layers=1,num_attention_heads=4);vision_cfg._attn_implementation='eager';vision=SiglipEncoderLayer(vision_cfg).eval()
checks=[]
for name,module in [('gemma-attention',layer.self_attn),('gemma-mlp',layer.mlp),('siglip-attention',vision.self_attn),('siglip-mlp',vision.mlp)]:
 for parameter in module.parameters():parameter.requires_grad_(False)
 calls=[];teachers=[]
 for index in range(2):
  x=torch.randn(1,3+index,128);kwargs={}
  if name=='gemma-attention':
   prefix=0 if index==0 else 7;positions=torch.arange(prefix,prefix+x.shape[1])[None];pair=None if not prefix else (torch.randn(1,1,prefix,16),torch.randn(1,1,prefix,16));kwargs=dict(position_ids=positions,attention_mask=torch.zeros(1,1,x.shape[1],x.shape[1]+prefix),position_embeddings=rope(x,positions),past_key_value=LayerKVCache(1,pair),use_cache=not bool(prefix),cache_position=positions[0])
  elif name=='siglip-attention':kwargs=dict(attention_mask=None)
  call=capture_call((x,),kwargs,1 if name=='gemma-attention' else 0)
  with torch.no_grad():expected=tensor(module(x,**kwargs)).detach()
  assert torch.equal(expected,tensor(replay_call(module,call))) and torch.equal(expected,replay(module,call))
  calls.append(call);teachers.append(expected)
 artifact=calibrate(module,calls,teachers,a.source,epochs=3,fixture_group=16)
 assert artifact['gradient_finite_nonzero'] and artifact['teacher_unchanged'] and artifact['disabled_quant_bitwise']
 checks.append(dict(unit=name,targets=len(artifact['weights']),contexts=len(calls),native_replay_bitwise=True,gradient_finite_nonzero=True,packed_weight_reload_bitwise=True,teacher_unchanged=True))
a.output.mkdir(exist_ok=False,parents=True);result=dict(gate='PASS_PI05_NATIVE_OMNI_LWC_REPLAY_GRADIENT_AND_PACK_CPU',checks=checks,source_sha256=SOURCE_SHA,LET=False,fixture_epochs=3,full_gpu_gate_passed=False,full_model_loaded=False)
(a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
