"""Original QVLA injector, protected components and strict gates-reload actions.

Sequentially restore actual base target weights from pinned safetensors shards,
then reapply a saved gate JSON. Never loads two GPU models or exports full weights.
"""
import argparse,hashlib,importlib.util,json,pathlib,sys
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--gates',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();r=a.root.resolve();a.output.mkdir(parents=True,exist_ok=False)
sys.path[:0]=[str(r/'code-cpu-prep-20261109-v1'),str(r/'sources/openvla-oft'),str(r/'sources/LIBERO')]
from qvla.action_jacobian_batch import initialize
from qvla.infra.common import digest,write_json
from experiments.robot import openvla_utils as utils
utils.update_auto_map=lambda _:None;utils.check_model_logic_mismatch=lambda _:None
cfg,model,head,proprio,processor=initialize(str(r/'models/openvla_oft'),7)
source=r/'sources/QVLA-official-oft-20261110-v1/openvla-oft/qvla/quantization.py';spec=importlib.util.spec_from_file_location('_qvla_original',source);q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
registered=q.target_modules(model);assign,metadata=q.load_gates(a.gates)
calibration=json.loads((r/'profiles/20261112-qvla-action4-v2/status.json').read_text());calls=calibration['calls'];active={n for n,c in calls.items() if c};inactive={n for n,c in calls.items() if not c}
assert set(calls)==set(registered) and set(assign)==active and len(inactive)==13 and not (active&inactive)
modules={n:registered[n] for n in active}
def th(v):return hashlib.sha256(v.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
def protected():
 result={}
 for prefix,m in [('model',model),('head',head),('proprio',proprio)]:
  for n,v in m.named_parameters():
   if prefix!='model' or not any(n==t+'.weight' for t in modules):result[prefix+'.'+n]=th(v)
 return result
before=protected();base_hash={n:th(m.weight) for n,m in modules.items()}
z=np.load(r/'runs/20261108-053-oft-bf16-admission-v3/episodes/task-00-reset-005/query-00000.npz',allow_pickle=False);obs={'full_image':z['image'].copy(),'wrist_image':z['wrist_image'].copy(),'state':z['state'].copy()};instruction=str(z['instruction'].item())
act=lambda:np.asarray(utils.get_vla_action(cfg,model,processor,{k:v.copy() for k,v in obs.items()},instruction,head,proprio))
assert np.array_equal(act(),z['raw_policy_chunk'])
count=q.apply_gates(model,a.gates);first=act();assert first.shape==(8,7) and np.isfinite(first).all();first_hash={n:th(m.weight) for n,m in modules.items()};assert any(first_hash[n]!=base_hash[n] for n in modules);assert protected()==before
saved=a.output/'gates-reloaded.json';saved.write_bytes(a.gates.read_bytes());assert digest(saved)==digest(a.gates)
from safetensors import safe_open
index=json.loads((r/'models/openvla_oft/model.safetensors.index.json').read_text())['weight_map'];files={}
for name in modules:
 key=name+'.weight';assert key in index;files.setdefault(index[key],[]).append((name,key))
with torch.no_grad():
 for file,names in files.items():
  with safe_open(r/'models/openvla_oft'/file,framework='pt',device='cpu') as sf:
   for name,key in names:modules[name].weight.copy_(sf.get_tensor(key).to(modules[name].weight.device))
assert all(th(m.weight)==base_hash[n] for n,m in modules.items());q.apply_gates(model,saved);second=act();assert np.array_equal(first,second) and all(th(m.weight)==first_hash[n] for n,m in modules.items());assert protected()==before
params=sum(m.weight.numel() for m in modules.values());weighted=sum(sum(bits)*modules[n].weight[0].numel() for n,bits in assign.items())/params
write_json(a.output/'result.json',dict(gate='PASS_QVLA_ORIGINAL_INJECTION_RELOAD_NATIVE_ACTION',gates_sha256=digest(a.gates),original_quantization_sha256=digest(source),quantized_modules=count,all16_matches_original_bf16=True,protected_components_unchanged=True,all_frozen=True,quantized_actions_finite=True,reload_actions_byte_equal=True,reload_all_target_weights_byte_equal=True,allocation_stats=metadata.get('stats'),actual_parameter_weighted_bits=weighted,active_targets=sorted(active),inactive_protected_targets=sorted(inactive),original_registered_targets=len(registered),trajectory_count=0))
print('PASS_QVLA_ORIGINAL_INJECTION_RELOAD_NATIVE_ACTION')
