from pathlib import Path
import json,sys,os,socket,subprocess,datetime,shutil
r=Path('/root/autodl-tmp/trajectory-drift-20261107');root=r/'control/20261117-084-pi05-resume-v1';reset=int(os.environ.get('VLA_PI05_RESET','15'));assert reset in [15,25,35,45];code=r/'code-084-native-runtime-20261117-v2';parent=r/'rounds/20261111-084-pi05-torch-independent-v2';rid=f'20261117-084-pi05-bf16-initial{reset}-v3';rd=root/'stages'/rid;sys.path.insert(0,str(code));os.chdir(code)
from qvla.infra.common import digest,canonical,write_json,now
from qvla.infra.plan import compile_plan,verify_plan
sys.path.append(str(root))
from guard_quota_window_v1 import bounded_quota_decision as quota_decision
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert shutil.disk_usage(r).free>=8*1024**3
q=json.loads((root/'quota.json').read_text());assert not (root/'CLOSE_NOW').exists();assert quota_decision(q,dict(quota_max_age_seconds=1020,close_remaining_percent=10,stop_expansion_remaining_percent=15))=='CONTINUE';authroot=json.loads((root/'authorization.json').read_text());assert socket.gethostname()==authroot['hostname'];assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==authroot['gpu_uuid'];assert json.loads((root/'guard-status.json').read_text())['state']=='CONTINUATION_ARMED'
oldplan=json.loads((parent/'plan.json').read_text());verify_plan(oldplan)
backend='openpi-pytorch-original-independent-h10-v2';existing=set()
for path in [*(r/'rounds').glob('*/case-runs/*/result.json'),*(r/'control').glob('*/stages/*/case-runs/*/result.json')]:
 z=json.loads(path.read_text())
 if z.get('backend')==backend and z.get('condition')=='bf16' and z.get('role')=='development' and z.get('gate','').startswith('PASS_PI05_TORCH_'):existing.add((z['suite'],z['task_id'],z['reset_index']))
assert all((s,0,5) in existing for s in ['libero_spatial','libero_object','libero_goal','libero_10'])
rd.mkdir(exist_ok=False)
for name in ['case-plans','case-runs','case-logs','audits','admission']:(rd/name).mkdir()
shutil.copyfile(parent/'admission/result.json',rd/'admission/result.json');assert digest(rd/'admission/result.json')==digest(parent/'admission/result.json')
old=json.loads((parent/'series.json').read_text());cases=[]
for source in [x for x in old['cases'] if x['role']=='development']:
 template=json.loads((parent/'case-plans'/(source['round_id']+'.json')).read_text())
 for task in range(10):
  if (source['suite'],task,reset) in existing:continue
  crid=f"{rid}-{source['suite']}-task{task}-reset{reset}";item=dict(template,round_id=crid,task_id=task,reset_index=reset,output=str(rd/'case-runs'/crid),registered_at=now());pp=rd/'case-plans'/(crid+'.json');write_json(pp,item);cases.append(dict(source,task_id=task,reset_index=reset,round_id=crid,template_sha256=digest(pp)))
assert cases
config=json.loads((parent/'config.json').read_text());config['control'].update(start_disk_gib=8,refresh_confirmation_wait_seconds=120)
for x in config['evaluation']['suites']:x.update(task_ids=list(range(10)),reset_offset=reset,resets_per_task=1)
b=config['binding'];b.pop('admission_argv',None);b.pop('post_evaluation_argv',None);b['evaluation_argv']=[str(rd) if x==str(parent) else x for x in b['evaluation_argv']];b['evaluation_argv']=[str(code/Path(x).relative_to(Path(oldplan['repo']))) if x.startswith(str(oldplan['repo'])+'/') else x for x in b['evaluation_argv']];proof=root/'native-runtime-v2-cpu-proof.json';b['gate_files'].append(dict(path=str(proof),sha256=digest(proof),expected_gate='PASS_PREPARED_FORK_SCIENCE_IDENTICAL_BOUNDED_QUOTA_RUNTIME_CPU'));b['gate_files'].append(dict(path=str(rd/'admission/result.json'),sha256=digest(rd/'admission/result.json'),expected_gate='PASS_PI05_TORCH_INDEPENDENT_BF16_RELOAD_INPUT'))
write_json(rd/'config.json',config);write_json(rd/'series.json',dict(old,cases=cases,control=config['control'],scope=f'Predeclared four suites task0..9 at initial{reset}; same backend, model, seed, cadence and noise; no calibration rerun'))
plan=compile_plan(config,code,rd)
for path,sha in oldplan['source_sha256'].items():
 rel=Path(path).relative_to(Path(oldplan['repo']))
 if rel.as_posix().startswith('qvla/infra/'):continue
 replacement=code/rel;assert digest(replacement)==sha;plan['source_sha256'][str(replacement)]=sha
for file in [code/'qvla/pi05_torch_backend_v2.py',code/'qvla/execute_pi05_torch_episode_v2.py']:
 if file.exists():plan['source_sha256'][str(file)]=digest(file)
plan['inputs']+=[dict(path=str(x),sha256=digest(x)) for x in [rd/'series.json',rd/'admission/result.json']]
plan.pop('plan_sha256');plan['plan_sha256']=canonical(plan);verify_plan(plan);write_json(rd/'plan.json',plan)
auth=dict(authroot,round_id=rid,shutdown_authorized=False,pause_only=True,config_sha256=canonical(config),expires_at=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=8)).isoformat(),source='Current084 continuation; missing same-backend reference cases, frozen seeds/noise/horizon/execution cadence; no Windows authority');write_json(rd/'authorization.json',auth);write_json(rd/'quota.json',q)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='4',PYTHONPATH=':'.join([str(code),str(r/'sources/openpi/src'),str(r/'sources/openpi/packages/openpi-client/src'),str(r/'sources/LIBERO')]))
argv=[sys.executable,'-u','-m','qvla.infra','run',str(rd/'plan.json'),'--authorization',str(rd/'authorization.json'),'--round-id',rid,'--quota',str(rd/'quota.json'),'--execute']
with (rd/'supervisor.log').open('xb') as f:p=subprocess.Popen(argv,cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
write_json(rd/'launch-receipt.json',dict(supervisor_pid=p.pid,plan_sha256=plan['plan_sha256'],started_at=now(),cases=len(cases)));write_json(root/'CURRENT_RUN.json',dict(stage=str(rd),condition='bf16',control_code=str(code)));print('MISSING_BF16_REFERENCES_LAUNCHED',len(cases),p.pid,plan['plan_sha256'])
