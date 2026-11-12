"""Original QVLA gate injection around the frozen OFT natural collector.

Preserves simulator/actions/seed cadence. Adds strict zero-quant first-input
comparison and profile metadata. Does not copy or alter frozen source files.
"""
import argparse,contextlib,hashlib,importlib,importlib.util,json,pathlib,sys,types
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--gates',type=pathlib.Path,required=True);p.add_argument('--reference',type=pathlib.Path,required=True);p.add_argument('--admission',type=pathlib.Path,required=True);p.add_argument('--suite',required=True);p.add_argument('--task-id',type=int,required=True);p.add_argument('--reset',type=int,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--round-id',required=True);a=p.parse_args()
r=a.root.resolve();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert json.loads(a.admission.read_text())['gate']=='PASS_QVLA_ORIGINAL_INJECTION_RELOAD_NATIVE_ACTION'
expected_gates=json.loads(a.admission.read_text())['gates_sha256'];assert sha(a.gates)==expected_gates
legacy=r/'code-cpu-prep-20261109-v1';sys.path[:0]=[str(legacy),str(r/'sources/openvla-oft'),str(r/'sources/LIBERO')]
src=legacy/'qvla/execute_oft_drift_v3.py';body=src.read_text()
old='model, action_head, proprio_projector, noisy_projector, processor = module.initialize_model(cfg)'
assert body.count(old)==1 and body.count('"condition": "bf16"')==1
body=body.replace(old,'model, action_head, proprio_projector, noisy_projector, processor = guarded_initialize(module, cfg)')
body=body.replace('"condition": "bf16"','"condition": "qvla_w4a16"').replace('"PASS_ACTUAL_OFT_BF16_NATURAL_TRACE"','"PASS_ACTUAL_OFT_QVLA_NATURAL_TRACE_PENDING_PAIRED_AUDIT"')
# Original native get_libero_env actually uses env_seed0; correct only NEW metadata.
body=body.replace('"env_seed": seed','"env_seed": 0')
source=r/'sources/QVLA-official-oft-20261110-v1/openvla-oft/qvla/quantization.py';spec=importlib.util.spec_from_file_location('_original_qvla_quantization',source);quant=importlib.util.module_from_spec(spec);spec.loader.exec_module(quant)
proof={}
def guarded_initialize(module,cfg):
 loaded=module.initialize_model(cfg);model,head,proprio,_,processor=loaded
 for m in [model,head,proprio]:m.requires_grad_(False)
 import torch
 assert all(not v.requires_grad for m in [model,head,proprio] for v in m.parameters())
 qfile=next(a.reference.glob('episodes/*/query-00000.npz'));z=np.load(qfile,allow_pickle=False)
 utils=importlib.import_module('experiments.robot.openvla_utils');obs={'full_image':z['image'].copy(),'wrist_image':z['wrist_image'].copy(),'state':z['state'].copy()}
 actual=np.asarray(utils.get_vla_action(cfg,model,processor,obs,str(z['instruction'].item()),head,proprio));assert np.array_equal(actual,z['raw_policy_chunk']),'Zero-quant action differs from same-case BF16'
 modules=quant.target_modules(model);gates,_=quant.load_gates(a.gates);ad=json.loads(a.admission.read_text());assert set(gates)==set(ad['active_targets']) and set(modules)==set(ad['active_targets'])|set(ad['inactive_protected_targets']),'QVLA active/unused scope differs'
 def th(v):return hashlib.sha256(v.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
 protected_inactive={name:th(modules[name].weight) for name in ad['inactive_protected_targets']}
 count=quant.apply_gates(model,a.gates);assert all(torch.isfinite(m.weight).all() for m in modules.values())
 assert all(th(modules[n].weight)==h for n,h in protected_inactive.items()),'Action-gradient-disconnected weights changed'
 proof.update(gate='PASS_QVLA_SAME_CASE_ZERO_QUANT_NATIVE_AND_INJECTION',first_query_sha256=sha(qfile),gates_sha256=expected_gates,quantized_modules=count,official_quantization_sha256=sha(source),base_and_heads_frozen=True,action_gradient_disconnected_targets_protected=True)
 return loaded
ns={'__name__':'_frozen_oft_qvla_binding','__file__':str(src),'guarded_initialize':guarded_initialize};exec(compile(body,str(src)+'[QVLA injection/metadata only]', 'exec'),ns)
observer=ns['observe_episode']
@contextlib.contextmanager
def observed(module,path,identity):
 identity.update(profile_sha256=expected_gates,qvla_original_source_commit='6d0a074459aa7efa09ef0cc86cfc339ce40d8b06',frozen_collector_sha256=sha(src),gate_budget_unit='mean output-channel bits',actual_env_seed=0)
 with observer(module,path,identity) as value:yield value
ns['observe_episode']=observed
args=argparse.Namespace(oft_source=r/'sources/openvla-oft',checkpoint=r/'models/openvla_oft',output=a.output,suite=a.suite,task_id=a.task_id,initial_state_index=a.reset,model_seed=7,round_id=a.round_id)
ns['run'](args)
(a.output/'qvla-injection-check.json').write_text(json.dumps(proof,indent=2)+'\n')
# Full same-first-observation pairing of actual quantized rollout, not just replay.
left=next(a.reference.glob('episodes/*/query-00000.npz'));right=next(a.output.glob('episodes/*/query-00000.npz'));l=np.load(left,allow_pickle=False);z=np.load(right,allow_pickle=False)
assert all(np.array_equal(l[k],z[k]) for k in ['image','wrist_image','state','instruction'])
(a.output/'paired-first-input.json').write_text(json.dumps(dict(gate='PASS_QVLA_ACTUAL_ROLLOUT_FIRST_INPUT_PAIRED',reference_query_sha256=sha(left),candidate_query_sha256=sha(right)),indent=2)+'\n')
