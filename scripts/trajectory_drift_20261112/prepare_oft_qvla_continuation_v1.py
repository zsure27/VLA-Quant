"""Fresh-host continuation: verify parent data, deduplicate, register only missing QVLA cases.

Does not start a GPU process. Existing plans, results and scientific sources are read-only.
"""
import argparse,datetime,json,os,pathlib,shutil,socket,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--code',type=pathlib.Path,required=True);p.add_argument('--request',type=pathlib.Path,required=True);a=p.parse_args()
r,code=a.root.resolve(),a.code.resolve();request=json.loads(a.request.read_text());root=r/'control'/request['round_id'];rd=root/'stages/qvla-remaining62-v1';parent=r/'control/20261112-061-oft-other-configs-v4';old=parent/'stages/qvla-complete80-v1';pilot=parent/'stages/qvla-pilot-v4'
assert socket.gethostname()==request['expected_hostname']
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert gpu==request['expected_gpu_uuid']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert shutil.disk_usage(r).free>=8*1024**3 and not root.exists()
sys.path.insert(0,str(code));os.chdir(code)
from qvla.infra.common import canonical,digest,write_json,now
from qvla.infra.plan import compile_plan,verify_plan
from qvla.infra.lifecycle import quota_decision
from qvla.baseline_contract import checkpoint_identity
config=json.loads((old/'config.json').read_text());quota=request['quota'];assert quota_decision(quota,config['control'])=='CONTINUE'
pin=json.loads((r/'checks/20261110-qvla-official-oft-source-v1.json').read_text());assert pin['commit']=='6d0a074459aa7efa09ef0cc86cfc339ce40d8b06'
for rel,sha in pin['members_sha256'].items():assert digest(r/'sources/QVLA-official-oft-20261110-v1'/rel)==sha
assert checkpoint_identity(r/'models/openvla_oft')['sha256']==config['model']['identity_sha256']
for stage in [old,pilot]:
 old_plan=json.loads((stage/'plan.json').read_text())
 for path,sha in old_plan['source_sha256'].items():assert digest(pathlib.Path(path))==sha,path
 for record in old_plan['inputs']:assert digest(pathlib.Path(record['path']))==record['sha256'],record['path']
 audits=json.loads((stage/'paired-audit/result.json').read_text());assert audits['gate'] in ['PASS_QVLA_RAW_QUERY_ACTION_POSE_VIDEO_AND_PAIRED_CASES','PASS_QVLA_AVAILABLE_RAW_PAIRS_QUOTA_STOP']
# Verify the stopped parent's actual raw files, including failures, against its saved manifest.
snapshot=r/'closure/20261112-061-oft-other-configs-v4-snapshot';manifest=json.loads((snapshot/'.closure/members.json').read_text());raw_verified=0
for row in manifest['members']:
 if row['path'].startswith('stages/') and '/case-runs/' in row['path']:
  f=parent/row['path'];assert f.stat().st_size==row['bytes'] and digest(f)==row['sha256'];raw_verified+=1
parent_rows=[]
for stage in [old,pilot]:
 for f in stage.glob('libero*-cases.json'):parent_rows+=json.loads(f.read_text())['completed']
done=set()
for row in parent_rows:
 assert digest(pathlib.Path(row['result_path']))==row['result_sha256'];case=row['case'];key=(case['suite'],case['task_id'],case['reset_index']);assert key not in done;done.add(key)
assert done=={tuple(x) for x in request['expected_completed_keys']} and len(done)==18
# Also scan legacy runs and every control round, rather than trusting a single latest index.
found=set();paths=set()
for pattern in ['*/case-runs/*/result.json','*/stages/*/case-runs/*/result.json']:
 paths.update((r/'control').glob(pattern))
if (r/'runs').exists():paths.update((r/'runs').rglob('result.json'))
for f in paths:
 row=json.loads(f.read_text())
 if row.get('model')=='openvla_oft' and row.get('condition')=='qvla_w4a16':
  assert row['gate']=='PASS_ACTUAL_OFT_QVLA_NATURAL_TRACE_PENDING_PAIRED_AUDIT'
  key=(row['suite'],row['task_id'],row['initial_state_index']);assert key not in found;found.add(key)
assert found==done,'New or conflicting completed QVLA cases: re-audit before registration'
series=json.loads((old/'series.json').read_text());all_cases=series['cases']+json.loads((pilot/'series.json').read_text())['cases'];assert len(all_cases)==80
remaining=[]
for case in all_cases:
 key=(case['suite'],case['task_id'],case['reset_index'])
 if key in done:continue
 assert digest(pathlib.Path(case['reference_result']))==case['reference_sha256']
 remaining.append(dict(case,id=rd.name+'-qvla-'+case['suite']+f"-task{case['task_id']}-reset{case['reset_index']}"))
assert len(remaining)==62 and len({(x['suite'],x['task_id'],x['reset_index']) for x in remaining})==62
rd.mkdir(parents=True)
for name in ['case-runs','case-logs','admission']:(rd/name).mkdir()
shutil.copyfile(old/'admission/result.json',rd/'admission/result.json');proof=json.loads((rd/'admission/result.json').read_text());assert proof['gate']=='PASS_QVLA_ORIGINAL_INJECTION_RELOAD_NATIVE_ACTION' and digest(pathlib.Path(series['gates']))==proof['gates_sha256']
for gate in config['binding']['gate_files']:
 assert digest(pathlib.Path(gate['path']))==gate['sha256']
config['binding'].pop('admission_argv',None);config['binding']['id']='qvla-same80-only62-missing-continuation-v1'
config['binding']['evaluation_argv']=[sys.executable,str(code/'scripts/trajectory_drift_20261112/run_oft_qvla_cases_v4.py'),'--root',str(r),'--round-dir',str(rd),'--suite','{suite}']
config['binding']['evaluation_audit_argv']=[sys.executable,str(code/'scripts/trajectory_drift_20261112/audit_oft_qvla_pairs_v2.py'),'--root',str(r),'--round-dir',str(rd),'--output',str(rd/'paired-audit')]
config['control'].update(start_disk_gib=8,backup_argv=[sys.executable,str(code/'scripts/infra_archive_run.py'),'--run-dir','{run_dir}','--prepare-only'])
config['scope'].update(actual_stage='62 missing QVLA paired trajectories; same80 cohort, skip18 QVLA and480 prior cases',parent_round=str(parent),source_runtime_commit=request['source_commit'])
series.update(cases=remaining,root_control=str(root),prior_complete_qvla_cases=18,prior_round=str(parent),only62missing=True);series.pop('only76missing',None)
preflight=dict(gate='PASS_QVLA_FRESH_HOST_SAME_RECIPE_18_DEDUP_62_MISSING',hostname=socket.gethostname(),gpu_uuid=gpu,old_raw_members_verified=raw_verified,prior_qvla_complete=18,remaining=62,parent_case_scan_results=len(paths),source_commit=pin['commit'],gates_sha256=proof['gates_sha256'],model_identity_sha256=config['model']['identity_sha256'],source_runtime_install=json.loads((r/'061-v5-runtime-install.json').read_text()),quota=quota)
write_json(root/'preflight.json',preflight);write_json(rd/'config.json',config);write_json(rd/'series.json',series);write_json(root/'quota.json',quota);write_json(rd/'quota.json',quota)
config['binding']['gate_files'].append(dict(path=str(root/'preflight.json'),sha256=digest(root/'preflight.json'),expected_gate=preflight['gate']));write_json(rd/'config.json',config)
plan=compile_plan(config,code,rd)
for f in [code/'scripts/trajectory_drift_20261112/collect_oft_qvla_case_v4.py',r/'code-cpu-prep-20261109-v1/qvla/execute_oft_drift_v3.py',r/'code-cpu-prep-20261109-v1/diagnostics/native_oft_trace_v1.py']:plan['source_sha256'][str(f)]=digest(f)
for f in [rd/'series.json',rd/'admission/result.json',pathlib.Path(series['gates'])]:plan['inputs'].append(dict(path=str(f),sha256=digest(f)))
plan.pop('plan_sha256');plan['plan_sha256']=canonical(plan);verify_plan(plan);write_json(rd/'plan.json',plan)
auth=dict(authorized=True,shutdown_authorized=False,pause_only=True,server_id='061',hostname=socket.gethostname(),gpu_uuid=gpu,config_sha256=canonical(config),round_id=rd.name,expires_at=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=6)).isoformat(),source='Current user reopened061 GPU and requests continuation under existing plan and quota closure; no Windows closure')
write_json(rd/'authorization.json',auth);write_json(root/'authorization.json',dict(authorized=True,shutdown_authorized=True,hostname=socket.gethostname(),gpu_uuid=gpu,round_id=root.name,control_code=str(code),source=auth['source']))
write_json(root/'CURRENT_RUN.json',dict(control_code=str(code),stage=str(rd),gpu_uuid=gpu,source_commit=pin['commit']));write_json(root/'status.json',dict(state='REGISTERED_NOT_STARTED',round_id=root.name,registered_at=now()))
print(json.dumps(dict(registered_not_started=True,root_control=str(root),stage=str(rd),remaining_cases=62,plan_sha256=plan['plan_sha256'],gpu_uuid=gpu)))
