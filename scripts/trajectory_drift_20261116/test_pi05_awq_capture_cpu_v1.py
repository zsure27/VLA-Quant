"""CPU fixture verifies capture/role/input gates, not a measured VLA policy."""
import argparse,os,json,hashlib,copy
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import torch,numpy as np
from transformers import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer
from qvla.pi05_awq_capture_block_v1 import collect,BACKEND_ID
torch.set_num_threads(1);torch.manual_seed(7);assert not torch.cuda.is_available();a.output.mkdir(exist_ok=False,parents=True)
cfg=SiglipVisionConfig(hidden_size=128,intermediate_size=256,num_hidden_layers=1,num_attention_heads=4);cfg._attn_implementation='eager'
class Fixture:
 def __init__(self):self._model=SiglipEncoderLayer(cfg).eval().requires_grad_(False)
 def infer(self,obs,noise):
  with torch.no_grad():x=self._model(torch.as_tensor(obs['observation/image']),attention_mask=None)[0]
  return {'actions':x[0,:10,:7].numpy().copy()}
policy=Fixture();rows=[]
for s in range(4):
 folder=a.output/f'fixture-suite{s}';folder.mkdir()
 (folder/'result.json').write_text(json.dumps(dict(role='calibration',task_id=0,reset_index=0,condition='bf16',backend=BACKEND_ID,suite=f'fixture-suite{s}',gate='PASS_PI05_TORCH_CPU_FIXTURE',real_vla_episode=False)))
 for i in range(8):
  obs={'observation/image':torch.randn(1,16,128).numpy(),'observation/wrist_image':np.zeros((2,2,3),np.uint8),'observation/state':np.zeros(8,np.float32),'prompt':'CPU fixture'};noise=np.zeros((10,32),np.float32)
  output=policy.infer(obs,noise)['actions'];path=folder/f'query-{i}.npz'
  np.savez_compressed(path,image=obs['observation/image'],wrist_image=obs['observation/wrist_image'],state=obs['observation/state'],prompt=np.array(obs['prompt']),policy_noise=noise,raw_policy_chunk=output)
  rows.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),suite=f'fixture-suite{s}',role='calibration',reset_index=0))
calls,features=collect(policy,policy._model,rows)
assert len(features)==6 and len(calls)==4 and all(len(v)==32 for v in calls.values())
assert features['self_attn.q_proj'] is features['self_attn.v_proj'] is features['self_attn.k_proj']
checks=['fixed32_native_call_capture','shared_qkv_input_verified_and_stored_once','whole_action_bitwise','teacher_unchanged']
for kind,bad,budget in [('development_role',[dict(r,role='development') for r in rows],2*1024**3),('duplicate_content',[rows[0]]*32,2*1024**3),('memory_budget',rows,1)]:
 try:collect(policy,policy._model,bad,max_cpu_bytes=budget)
 except ValueError:checks.append(kind+'_rejected')
 else:raise AssertionError(kind+' accepted')
assert all(not m._forward_pre_hooks for m in policy._model.modules())
out=dict(gate='PASS_FIXED_ROLE_NATIVE_BLOCK_CALIBRATION_CAPTURE_CPU_FIXTURE',checks=checks,real_vla_model=False,formal_episodes=0,full_awq_admitted=False)
(a.output/'result.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
