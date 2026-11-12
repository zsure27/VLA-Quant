"""CPU blueprint for already-planned missing initial25 BF16 references; no launch."""
import argparse,json,pathlib,hashlib,sys,os
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--control',type=pathlib.Path,required=True);a=p.parse_args();r=a.root.resolve();out=a.control/'cpu-preparation/bf16-initial25-blueprint-v1';out.mkdir(parents=True,exist_ok=False);os.environ['CUDA_VISIBLE_DEVICES']='';sys.path[:0]=[str(r/'code-cpu-prep-20261109-v1'),str(r/'sources/LIBERO')]
from libero.libero import benchmark
from qvla.infra.common import digest
receipt=digest(r/'models/openvla_oft/DOWNLOAD_VERIFICATION.json');found={};paths=set((r/'runs').rglob('result.json')) if (r/'runs').exists() else set()
for pattern in ['*/case-runs/*/result.json','*/stages/*/case-runs/*/result.json']:paths.update((r/'control').glob(pattern))
for f in paths:
 if a.control.resolve() in f.resolve().parents:continue  # Never read the active QVLA cases.
 z=json.loads(f.read_text())
 if z.get('model')=='openvla_oft' and z.get('condition')=='bf16' and z.get('initial_state_index')==25 and z.get('checkpoint_receipt_sha256')==receipt and z.get('gate')=='PASS_ACTUAL_OFT_BF16_NATURAL_TRACE':
  key=(z['suite'],z['task_id'],25);assert key not in found;found[key]=dict(result=str(f),sha256=digest(f))
cases=[]
for suite in ['libero_spatial','libero_object','libero_goal','libero_10']:
 b=benchmark.get_benchmark_dict()[suite]()
 for task in range(10):
  key=(suite,task,25)
  if key in found:continue
  state=np.asarray(b.get_task_init_states(task)[25]);assert np.isfinite(state).all()
  cases.append(dict(suite=suite,task_id=task,reset_index=25,condition='bf16',initial_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),initial_state_shape=list(state.shape),initial_state_dtype=str(state.dtype)))
assert len(cases)+len(found)==40
legacy=r/'code-cpu-prep-20261109-v1';sources={str(legacy/x):digest(legacy/x) for x in ['qvla/execute_oft_drift_v3.py','qvla/on_policy_capture.py','qvla/baseline_contract.py','qvla/trajectory_diagnostics_v1.py','diagnostics/native_oft_trace_v1.py']}
result=dict(gate='PASS_CPU_BF16_INITIAL25_REFERENCE_BLUEPRINT_NOT_REGISTERED',cases=cases,already_complete=found if not found else list(found.values()),missing=len(cases),preregistered_target=40,role='development shared BF16 references for originally planned reset5/15/25/35/45; no quantized comparisons until paired candidates complete',checkpoint_receipt_sha256=receipt,external_frozen_sources=sources,model_seed=7,actual_native_env_seed=0,legacy_env_seed_label=7,actions_per_query=8,no_gpu_started=True,activation_requires='Current QVLA62 raw audit/terminal gate, fresh quota, idle GPU, disk>=8GiB and new per-stage authorization')
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['cases','external_frozen_sources']}),flush=True)
