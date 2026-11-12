"""At a terminal QVLA boundary, register/start only missing preplanned initial25 references."""
import argparse,datetime,json,os,pathlib,shutil,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--control',type=pathlib.Path,required=True);a=p.parse_args();r=a.root.resolve();root=a.control.resolve();current=json.loads((root/'CURRENT_RUN.json').read_text());old=pathlib.Path(current['stage']);code=pathlib.Path(current['control_code']);rd=root/'stages/bf16-initial25-v1'
assert old.name.startswith('qvla-remaining') and json.loads((old/'status.json').read_text())['state']=='PAUSED_SERVER_LEFT_ON' and json.loads((old/'status.json').read_text())['reason']=='task_commands_complete'
proof=old/'paired-audit/result.json';gate=json.loads(proof.read_text());assert gate['gate']=='PASS_QVLA_RAW_QUERY_ACTION_POSE_VIDEO_AND_PAIRED_CASES' and gate['completed']==len(json.loads((old/'series.json').read_text())['cases'])
parent_series=json.loads((old/'series.json').read_text());assert parent_series.get('prior_complete_qvla_cases',18)+gate['completed']==80
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip() and shutil.disk_usage(r).free>=8*1024**3 and not (root/'STOP_NEW_GPU').exists()
sys.path.insert(0,str(code));os.chdir(code)
from qvla.infra.common import canonical,digest,write_json,now
from qvla.infra.plan import compile_plan,verify_plan
from qvla.infra.lifecycle import quota_decision
blueprint_path=root/'cpu-preparation/bf16-initial25-blueprint-v1/result.json';blueprint=json.loads(blueprint_path.read_text());assert blueprint['gate']=='PASS_CPU_BF16_INITIAL25_REFERENCE_BLUEPRINT_NOT_REGISTERED' and blueprint['missing']==40
config=json.loads((old/'config.json').read_text());quota=json.loads((root/'quota.json').read_text());assert quota_decision(quota,config['control'])=='CONTINUE'
runtime_auth=json.loads((root/'authorization.json').read_text());gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert gpu==runtime_auth['gpu_uuid']
for path,sha in blueprint['external_frozen_sources'].items():assert digest(pathlib.Path(path))==sha
case_paths=set((r/'runs').rglob('result.json'))
for pattern in ['*/case-runs/*/result.json','*/stages/*/case-runs/*/result.json']:case_paths.update((r/'control').glob(pattern))
already=set()
for path in case_paths:
 z=json.loads(path.read_text())
 if z.get('condition')=='bf16' and z.get('model')=='openvla_oft' and z.get('initial_state_index')==25 and z.get('checkpoint_receipt_sha256')==blueprint['checkpoint_receipt_sha256']:already.add((z['suite'],z['task_id'],25))
assert not already,'Rebuild blueprint if any case has completed since preparation'
rd.mkdir()
for name in ['case-runs','case-logs']:(rd/name).mkdir()
cases=[]
for case in blueprint['cases']:cases.append(dict(case,id=rd.name+'-'+case['suite']+f"-task{case['task_id']}-reset25"))
old_series=json.loads((old/'series.json').read_text());reference=json.loads(pathlib.Path(old_series['cases'][0]['reference_result']).read_text())
series=dict(cases=cases,root_control=str(root),external_frozen_sources=blueprint['external_frozen_sources'],checkpoint_receipt_sha256=blueprint['checkpoint_receipt_sha256'],expected_oft_source_commit=reference['oft_source_commit'],actual_native_env_seed=0,legacy_env_seed_label=7,preregistered_initials=[5,15,25,35,45],role='new unpaired BF16 shared references; original five-initial plan; no candidate score denominator expansion')
config['evaluation']['suites']=[dict(name=s,role='development',reset_offset=25,resets_per_task=1) for s in ['libero_spatial','libero_object','libero_goal','libero_10']]
config['binding'].pop('admission_argv',None);config['binding']['id']='native-bf16-original-collector-initial25-shared-reference-v1';config['binding']['evaluation_argv']=[sys.executable,str(code/'scripts/trajectory_drift_20261112/run_bf16_reference_cases_v1.py'),'--root',str(r),'--round-dir',str(rd),'--suite','{suite}'];config['binding']['evaluation_audit_argv']=[sys.executable,str(code/'scripts/trajectory_drift_20261112/audit_bf16_reference_cases_v1.py'),'--root',str(r),'--round-dir',str(rd)]
config['binding']['gate_files'].append(dict(path=str(proof),sha256=digest(proof),expected_gate=gate['gate']));config['scope']=dict(actual_stage='40 new unpaired BF16 reference cases at originally planned initial25; same native recipe, seed and execution cadence',parent_qvla_stage=str(old),current_qvla_same80_complete=True,model_seed=7,actual_env_seed=0,no_quantized_comparative_score_for_initial25_yet=True)
write_json(rd/'series.json',series);write_json(rd/'config.json',config);write_json(rd/'quota.json',quota)
plan=compile_plan(config,code,rd)
for path,sha in blueprint['external_frozen_sources'].items():plan['inputs'].append(dict(path=path,sha256=sha,role='external frozen native scientific source'))
for file in [rd/'series.json',blueprint_path]:plan['inputs'].append(dict(path=str(file),sha256=digest(file)))
plan.pop('plan_sha256');plan['plan_sha256']=canonical(plan);verify_plan(plan);write_json(rd/'plan.json',plan)
auth=json.loads((old/'authorization.json').read_text());auth.update(round_id=rd.name,config_sha256=canonical(config),source='Current opened061 continuation of original four-suite/10-task/five-initial study; shared reference initial25 while optional methods remain gated');write_json(rd/'authorization.json',auth)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',PYTHONPATH=str(code),MUJOCO_GL='egl',TOKENIZERS_PARALLELISM='false');args=[sys.executable,'-u','-m','qvla.infra','run',str(rd/'plan.json'),'--authorization',str(rd/'authorization.json'),'--round-id',rd.name,'--quota',str(rd/'quota.json'),'--execute']
with (rd/'supervisor.log').open('xb') as f:proc=subprocess.Popen(args,cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
rec=dict(supervisor_pid=proc.pid,stage=str(rd),started_at=now(),new_reference_cases=40,initial_state_index=25,plan_sha256=plan['plan_sha256'],candidate_pairs=0);write_json(rd/'launch-receipt.json',rec);current['stage']=str(rd);write_json(root/'CURRENT_RUN.json',current);print(json.dumps(rec),flush=True)
