from pathlib import Path
import json,sys,os,shutil
r=Path('/root/autodl-tmp/trajectory-drift-20261107');root=r/'control/20261117-084-pi05-resume-v1';old=r/'control/20261116-084-pi05-resume-v2';code=r/'code-084-native-runtime-20261117-v2';rd=root/'stages/20261117-084-pi05-awq-calibration-v3'
sys.path.insert(0,str(code));os.chdir(code)
from qvla.infra.common import digest,canonical,write_json
from qvla.infra.plan import compile_plan,verify_plan
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and shutil.disk_usage(r).free>=12*1024**3
cfg=json.loads((r/'rounds/20261111-084-pi05-torch-independent-v2/config.json').read_text())
cfg['task']['changed_variable']='Pinned original AWQ W4A16 native-block calibration only; zero formal evaluation episodes'
cfg['quantization']=dict(method='awq',weight_bits=4,activation_bits=16)
cfg['control'].update(start_disk_gib=12,runtime_disk_gib=6,refresh_confirmation_wait_seconds=120)
cfg['control'].pop('weekly_close_remaining_percent',None)
selection=old/'cpu-preparation/awq-selected32-inputs-v1/index.json';selected=json.loads(selection.read_text());cfg['data']['calibration']=selected['records']
capture=r/'rounds/20261111-084-pi05-smoothquant-a8-v2/capture'
gates=[(selection,selected['gate']),(capture/'result.json','PASS_PI05_414_CALIBRATION_REPLAY_STATS'),(old/'stages/20261116-084-pi05-bf16-initial25-v3/native-block-diagnostic/result.json','PASS_ACTUAL_PI05_NATIVE_BLOCK_REPLAY_CALIBRATION'),(root/'cpu-awq-variable-context-v3/result.json','PASS_NATIVE_AWQ_VARIABLE_CONTEXT_BATCH_AND_FULL_MODULE_SEARCH_CPU'),(root/'cpu-awq-block-integration-v3/result.json','PASS_NATIVE_BLOCK_AWQ_SEARCH_FOLD_CLIP_PACK_RELOAD_CPU'),(root/'cpu-awq-mixed-native-v1/result.json','PASS_NATIVE_MIXED_BF16_FP32_BLOCK_AWQ_RELOAD_CPU'),(root/'cpu-awq-profile-load-v1/result.json','PASS_AWQ_63_TINY_NATIVE_BLOCK_PROFILE_LOAD_CONTRACT_CPU'),(root/'native-runtime-v2-cpu-proof.json','PASS_PREPARED_FORK_SCIENCE_IDENTICAL_BOUNDED_QUOTA_RUNTIME_CPU')]
layout=root/'cpu-awq-cache-layout-v2/result.json';gates.append((layout,'PASS_AWQ_CACHE_LAYOUT_VALUES_AND_QUANT_ARTIFACTS_BITWISE_CPU'))
for path,gate in gates:assert json.loads(path.read_text())['gate']==gate
b=cfg['binding'];b.update(id='pi05-native-original-awq-calibration-only-v1',quantizers=['awq'],gate_files=[dict(path=str(p),sha256=digest(p),expected_gate=g) for p,g in gates],admission_argv=['{python}','{repo}/scripts/trajectory_drift_20261116/calibrate_pi05_awq_blocks_v2.py','--root',str(r),'--capture',str(capture),'--manifest32',str(selection),'--output',str(rd/'profile')],evaluation_argv=['{python}','-c',"raise RuntimeError('Calibration-only config: no development evaluation is admitted here')"])
legacy=root/'stages/20261117-084-pi05-awq-calibration-v2';legacy_progress=legacy/'profile/progress.json';lz=json.loads(legacy_progress.read_text());assert len(lz['entries'])==18 and sum(x['targets'] for x in lz['entries'])==126
assert json.loads((legacy/'status.json').read_text())['stages'][0]['exit_code']==1
assert 'Fixed calibration exceeds current-block memory budget' in (legacy/'awq-calibration-and-full-policy-admission.stdout.log').read_text()
b['admission_argv'] += ['--reuse-completed',str(legacy/'profile'),'--layout-proof',str(layout)]
for name in ['post_evaluation_argv','evaluation_audit_argv','post_training_audit_argv']:b.pop(name,None)
rd.mkdir(exist_ok=False);write_json(rd/'config.json',cfg)
plan=compile_plan(cfg,code,rd)
# Explicit calibration-only plan. Suites above declare prospective evaluation
# identities solely for data-role separation; the fail-closed placeholder is
# never scheduled. A new profile-bound plan is required for any trajectories.
plan['stages']=[s for s in plan['stages'] if s['id']=='model-admission'];assert len(plan['stages'])==1
plan['stages'][0]['id']='awq-calibration-and-full-policy-admission'
plan['execution_scope']='CALIBRATION_ONLY_ZERO_FORMAL_EPISODES'
for file in [*code.glob('qvla/pi05_awq*.py'),code/'qvla/pi05_block_replay_v2.py',code/'qvla/pi05_packed_weight_v1.py',code/'qvla/pi05_torch_backend_v2.py']:
 plan['source_sha256'][str(file)]=digest(file)
for file in [capture/'statistics.pt',r/'sources/openpi/src/openpi/models_pytorch/pi0_pytorch.py',r/'sources/openpi/src/openpi/models_pytorch/gemma_pytorch.py',*(r/'sources/quant-source-sparse-v2/llm-awq/awq/quantize').glob('*.py')]:plan['inputs'].append(dict(path=str(file),sha256=digest(file)))
import inspect,transformers,torch
from transformers.models.gemma import modeling_gemma
from transformers.models.siglip import modeling_siglip
from transformers.models.paligemma import modeling_paligemma
from transformers import cache_utils
for module in [modeling_gemma,modeling_siglip,modeling_paligemma,cache_utils]:
 file=Path(inspect.getfile(module));plan['inputs'].append(dict(path=str(file),sha256=digest(file)))
for rel in ['openpi/policies/policy_config.py','openpi/policies/policy.py','openpi/transforms.py']:
 file=r/'sources/openpi/src'/rel;plan['inputs'].append(dict(path=str(file),sha256=digest(file)))
write_json(rd/'RUNTIME_DEPENDENCIES.json',dict(torch_version=torch.__version__,transformers_version=transformers.__version__,external_python_inputs=[x for x in plan['inputs'] if 'site-packages' in x['path']]))
plan['inputs'].append(dict(path=str(rd/'RUNTIME_DEPENDENCIES.json'),sha256=digest(rd/'RUNTIME_DEPENDENCIES.json')))
for file in [legacy_progress,legacy/'status.json',legacy/'execution-plan-v1.json']:
 plan['inputs'].append(dict(path=str(file),sha256=digest(file)))
for row in lz['entries']:
 file=legacy/'profile'/row['file'];assert digest(file)==row['sha256'];plan['inputs'].append(dict(path=str(file),sha256=row['sha256']))
plan.pop('plan_sha256');plan['plan_sha256']=canonical(plan);verify_plan(plan);write_json(rd/'plan.json',plan)
write_json(rd/'CPU_PLAN_READY.json',dict(gate='PASS_ORIGINAL_AWQ_STREAMED_CALIBRATION_CPU_PLAN',plan_sha256=plan['plan_sha256'],gpu_not_started=True,formal_episodes=0,fresh_quota_gpu_idle_and_previous_full_audit_required=True))
print('AWQ_CALIBRATION_CPU_PLAN_READY',plan['plan_sha256'])
