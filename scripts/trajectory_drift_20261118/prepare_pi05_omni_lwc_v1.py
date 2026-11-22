"""CPU-only registration of actual OmniQuant LWC smoke/full calibration stages."""
import argparse,json,sys
from pathlib import Path
from qvla.infra.common import canonical,digest,write_json
from qvla.infra.plan import compile_plan,verify_plan
import pi05_quant_catalog_20261118_v1

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--control',type=Path,required=True);p.add_argument('--code',type=Path,required=True);p.add_argument('--phase',choices=['smoke','full'],required=True);a=p.parse_args()
r,control,code=a.root.resolve(),a.control.resolve(),a.code.resolve()
parent=r/'control/20261117-084-pi05-other-methods-v1/stages/20261117-084-other-awq-calibration-v1'
previous=json.loads((parent/'plan.json').read_text());verify_plan(previous)
cpu=control/'cpu-omni-lwc-v2/result.json';proof=json.loads(cpu.read_text())
assert proof['gate']=='PASS_PI05_NATIVE_OMNI_LWC_REPLAY_GRADIENT_AND_PACK_CPU'
adapter=code/'qvla/pi05_omni_lwc_units_v1.py';assert digest(adapter)==proof['adapter_sha256']
stage=control/'stages'/('20261118-084-omni-smoke6-v1' if a.phase=='smoke' else '20261118-084-omni-full126-v1')
config=json.loads((parent/'config.json').read_text());identity=code/'Ex_Reports/_tasks'/config['task']['id']/'task.json'
if identity.exists():config['task']=json.loads(identity.read_text())
config['quantization']=dict(method='omniquant',weight_bits=4,activation_bits=16)
capture=r/'rounds/20261111-084-pi05-smoothquant-a8-v2/capture'
manifest=r/'control/20261116-084-pi05-resume-v2/cpu-preparation/awq-selected32-inputs-v1/index.json'
assert manifest.exists()
binding=config['binding'];binding.update(id='pi05-original-omni-lwc-native-units-v1',quantizers=['omniquant'])
binding['gate_files'].append(dict(path=str(cpu),sha256=digest(cpu),expected_gate=proof['gate']))
if a.phase=='full':
 smoke=control/'stages/20261118-084-omni-smoke6-v1/profile/result.json';assert json.loads(smoke.read_text())['gate']=='PASS_PI05_OMNI_LWC_NATIVE_SIX_UNITS_GPU_SMOKE'
 binding['gate_files'].append(dict(path=str(smoke),sha256=digest(smoke),expected_gate='PASS_PI05_OMNI_LWC_NATIVE_SIX_UNITS_GPU_SMOKE'))
binding['admission_argv']=['{python}','{repo}/scripts/trajectory_drift_20261118/calibrate_pi05_omni_lwc_v1.py','--root',str(r),'--capture',str(capture),'--manifest32',str(manifest),'--cpu-proof',str(cpu),'--output',str(stage/'profile')]+(['--smoke-only'] if a.phase=='smoke' else [])
binding['evaluation_argv']=['{python}','-c',"raise RuntimeError('No formal trajectories before full OmniQuant GPU admission')"]
for key in ['post_evaluation_argv','evaluation_audit_argv','post_training_audit_argv']:binding.pop(key,None)
config['control'].update(start_disk_gib=12,runtime_disk_gib=6,refresh_confirmation_wait_seconds=120)
stage.mkdir(exist_ok=False);write_json(stage/'config.json',config)
plan=compile_plan(config,code,stage);plan['stages']=[s for s in plan['stages'] if s['id']=='model-admission'];assert len(plan['stages'])==1
plan['stages'][0]['id']='omni-native-six-unit-smoke' if a.phase=='smoke' else 'omni-native-full-unit-calibration-and-admission'
plan['execution_scope']='CALIBRATION_ONLY_ZERO_FORMAL_EPISODES';plan['source_sha256'].update(previous['source_sha256']);plan['inputs']+=previous['inputs']
for file in [adapter,code/'pi05_quant_catalog_20261118_v1.py',Path(__file__).resolve()]:plan['source_sha256'][str(file)]=digest(file)
for rel,sha in proof['source_sha256'].items():
 file=r/'sources/OmniQuant-archive-v1'/rel;assert digest(file)==sha;plan['inputs'].append(dict(path=str(file),sha256=sha))
plan['inputs'] += [dict(path=str(capture/'statistics.pt'),sha256=digest(capture/'statistics.pt')),dict(path=str(manifest),sha256=digest(manifest))]
plan.pop('plan_sha256');plan['plan_sha256']=canonical(plan);verify_plan(plan);write_json(stage/'plan.json',plan)
write_json(stage/'CPU_PLAN_READY.json',dict(gate='PASS_ORIGINAL_OMNI_LWC_CALIBRATION_CPU_PLAN',plan_sha256=plan['plan_sha256'],gpu_started=False,formal_episodes=0,phase=a.phase,native_units=6 if a.phase=='smoke' else 126,epochs=2 if a.phase=='smoke' else 20,LET=False))
print(json.dumps(dict(stage=str(stage),plan_sha256=plan['plan_sha256'],phase=a.phase,gpu_started=False)))
