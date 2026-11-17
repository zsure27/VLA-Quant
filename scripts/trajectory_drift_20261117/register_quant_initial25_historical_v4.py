from pathlib import Path
import json,sys,os,socket,subprocess,datetime,shutil
r=Path('/root/autodl-tmp/trajectory-drift-20261107');root=r/'control/20261117-084-pi05-resume-v1';oldroot=r/'control/20261116-084-pi05-resume-v1';reset=int(os.environ.get('VLA_PI05_RESET','15'));assert reset in [15,25,35,45];condition=os.environ.get('VLA_PI05_CONDITION','smoothquant_w4a8');assert condition in ['smoothquant_w4a8','smoothquant_w4a4','gptq_w4a16'];bits=int(condition.rsplit('a',1)[1]);code=r/('code-084-cpu-prepared-20261112-v1' if condition=='gptq_w4a16' else 'code-084-pi05-smoothquant-v2');parent=r/'rounds/20261111-084-pi05-smoothquant-a8-v2';refstage=root/'stages'/f'20261117-084-pi05-bf16-initial{reset}-v2';refstage=r/'control/20261116-084-pi05-resume-v2/stages/20261116-084-pi05-bf16-initial25-v3' if reset==25 else refstage;attempt=os.environ.get('VLA_PI05_ATTEMPT','v2');assert attempt in ['v2','v3','v4'];rid=f'20261117-084-pi05-{condition}-initial{reset}-{attempt}';rd=root/'stages'/rid;sys.path.insert(0,str(code));os.chdir(code)
if condition=='gptq_w4a16':import pi05_gptq_catalog_20261116_v1
deny=root/'EXCLUDED_FUTURE_CONFIGURATIONS_V1.json'
if deny.exists():
 exclusion=json.loads(deny.read_text());assert not (condition==exclusion['condition'] and reset in exclusion['excluded_reset_indices']),'Explicit user cancellation: do not launch this condition/reset'
sys.path.append(str(root))
from qvla.infra.common import digest,canonical,write_json,now
from qvla.infra.plan import compile_plan,verify_plan
from guard_quota_window_v1 import bounded_quota_decision as quota_decision
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert shutil.disk_usage(r).free>=8*1024**3
state=json.loads((refstage/'status.json').read_text());assert state['state']=='PAUSED_SERVER_LEFT_ON' and all(x['exit_code']==0 for x in state['stages'])
q=json.loads((root/'quota.json').read_text());assert not (root/'CLOSE_NOW').exists();assert quota_decision(q,dict(quota_max_age_seconds=1020,close_remaining_percent=10,stop_expansion_remaining_percent=15))=='CONTINUE';authroot=json.loads((root/'authorization.json').read_text());assert socket.gethostname()==authroot['hostname'];assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==authroot['gpu_uuid'];assert json.loads((root/'guard-status.json').read_text())['state']=='CONTINUATION_ARMED'
analysis=Path(os.environ['VLA_PREDECESSOR_ANALYSIS']);az=json.loads(analysis.read_text());assert az['gate']=='PASS_AVAILABLE_PI05_SAME_BACKEND_RAW_GEOMETRY_AUDIT' and az['reset_indices']==[reset];assert az['coverage']['bf16']['complete']==40
if condition=='gptq_w4a16':assert az['coverage']['smoothquant_w4a8']['complete']==40
profile=(oldroot/'stages/20261116-084-pi05-gptq-pilot-v4/profile/profile.pt') if condition=='gptq_w4a16' else parent/'profile/profile.pt';pr=json.loads(profile.with_name('result.json').read_text());assert pr['precision_gates'][str(bits)]=='PASS_RELOAD_FINITE' and digest(profile)==pr['profile_sha256']
backend='openpi-pytorch-original-independent-h10-v2';refs={};existing=set()
for path in [*(r/'rounds').glob('*/case-runs/*/result.json'),*(r/'control').glob('*/stages/*/case-runs/*/result.json')]:
 z=json.loads(path.read_text())
 if z.get('backend')!=backend or z.get('role')!='development' or not z.get('gate','').startswith('PASS_PI05_TORCH_'):continue
 k=(z['suite'],z['task_id'],z['reset_index'])
 if z.get('condition')=='bf16':
  assert k not in refs,'Duplicate same-backend BF16 reference';refs[k]=path
 if z.get('condition')==condition:existing.add(k)
assert len([k for k in refs if k[2]==reset])==40
old=json.loads((parent/'series.json').read_text());rd.mkdir(exist_ok=False)
for name in ['case-plans','case-runs','case-logs','audits']:(rd/name).mkdir()
cases=[];reference_rows=[]
for source in old['cases']:
 template=json.loads((parent/'case-plans'/(source['round_id']+'.json')).read_text())
 for task in range(10):
  k=(source['suite'],task,reset)
  if k in existing:continue
  reference=refs[k];crid=f"{rid}-{source['suite']}-task{task}-reset{reset}";item=dict(template,round_id=crid,task_id=task,reset_index=reset,condition=condition,bf16_reference=str(reference.parent),output=str(rd/'case-runs'/crid),registered_at=now())
  if condition=='gptq_w4a16':item['runner']='qvla.execute_pi05_torch_gptq_episode_v1'
  pp=rd/'case-plans'/(crid+'.json');write_json(pp,item);cases.append(dict(source,task_id=task,reset_index=reset,round_id=crid,template_sha256=digest(pp)));reference_rows.append(dict(path=str(reference),sha256=digest(reference)))
assert cases
write_json(rd/'reference-gate.json',dict(gate='PASS_SAME_BACKEND_40_REFERENCE_COVERAGE',condition=condition,new_missing=len(cases),references=reference_rows))
config=json.loads((parent/'config.json').read_text());config['control'].update(start_disk_gib=8,refresh_confirmation_wait_seconds=120);config['quantization']=dict(method='gptq' if condition=='gptq_w4a16' else 'smoothquant',weight_bits=4,activation_bits=bits,profiles=[dict(path=str(profile),sha256=digest(profile))])
for x in config['evaluation']['suites']:x.update(task_ids=list(range(10)),reset_offset=reset,resets_per_task=1)
b=config['binding'];b.pop('admission_argv',None);b.update(id='pi05-'+condition+'-first40-v1',quantizers=[config['quantization']['method']]);b['gate_files']=[b['gate_files'][0],dict(path=str(profile.with_name('result.json')),sha256=digest(profile.with_name('result.json')),expected_gate=pr['gate']),dict(path=str(rd/'reference-gate.json'),sha256=digest(rd/'reference-gate.json'),expected_gate='PASS_SAME_BACKEND_40_REFERENCE_COVERAGE')]
entry='run_pi05_gptq_cases_v1.py' if condition=='gptq_w4a16' else 'run_pi05_sq_cases_v2.py';b['evaluation_argv']=['{python}','{repo}/scripts/trajectory_drift_20261111/'+entry,'--root',str(r),'--round-dir',str(rd),'--suite','{suite}','--role','development','--activation-bits',str(bits)]
write_json(rd/'config.json',config);write_json(rd/'series.json',dict(old,cases=cases,control=config['control'],profile=str(profile),scope=f'Predeclared four suites task0..9 at initial{reset}, frozen model/profile/seed/cadence/noise; exact same-initial BF16 references'))
plan=compile_plan(config,code,rd)
for rel in ['qvla/pi05_torch_backend_v2.py','qvla/execute_pi05_torch_gptq_episode_v1.py' if condition=='gptq_w4a16' else 'qvla/execute_pi05_torch_sq_episode_v2.py','scripts/trajectory_drift_20261111/'+entry]:plan['source_sha256'][str(code/rel)]=digest(code/rel)
if condition=='gptq_w4a16':plan['source_sha256'][str(code/'pi05_gptq_catalog_20261116_v1.py')]=digest(code/'pi05_gptq_catalog_20261116_v1.py')
plan['inputs']+=[dict(path=str(analysis),sha256=digest(analysis)),dict(path=str(rd/'series.json'),sha256=digest(rd/'series.json')),*reference_rows];plan.pop('plan_sha256');plan['plan_sha256']=canonical(plan);verify_plan(plan);write_json(rd/'plan.json',plan)
auth=dict(authroot,round_id=rid,shutdown_authorized=False,pause_only=True,config_sha256=canonical(config),expires_at=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=8)).isoformat(),source='Current084 continuation, remaining first40 same-backend paired cases; no Windows shutdown authority');write_json(rd/'authorization.json',auth);write_json(rd/'quota.json',q)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='4',PYTHONPATH=':'.join([str(code),str(r/'sources/openpi/src'),str(r/'sources/openpi/packages/openpi-client/src'),str(r/'sources/LIBERO')]))
argv=[sys.executable,'-u','-m','qvla.infra'];argv+=['--plugin','pi05_gptq_catalog_20261116_v1'] if condition=='gptq_w4a16' else [];argv+=['run',str(rd/'plan.json'),'--authorization',str(rd/'authorization.json'),'--round-id',rid,'--quota',str(rd/'quota.json'),'--execute']
with (rd/'supervisor.log').open('xb') as f:p=subprocess.Popen(argv,cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
write_json(rd/'launch-receipt.json',dict(supervisor_pid=p.pid,plan_sha256=plan['plan_sha256'],started_at=now(),cases=len(cases)));write_json(root/'CURRENT_RUN.json',dict(stage=str(rd),condition=condition,control_code=str(code)));print('MISSING_QUANTIZED_CASES_LAUNCHED',condition,len(cases),p.pid,plan['plan_sha256'])
