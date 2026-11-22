"""Register missing SQ W4A8 / GPTQ W4A16 cases, CPU only, no recipe change."""
import argparse,datetime,importlib,json,socket
from pathlib import Path
from qvla.infra.common import digest,canonical,write_json,now
from qvla.infra.plan import compile_plan,verify_plan
from qvla.audit_pi05_torch_trace_v2 import audit as audit_bf16

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--control',type=Path,required=True);p.add_argument('--code',type=Path,required=True);p.add_argument('--condition',choices=['smoothquant_w4a8','gptq_w4a16'],required=True);p.add_argument('--reset-index',type=int,choices=[35,45],required=True);p.add_argument('--round-id',required=True);a=p.parse_args()
r,c,code=a.root.resolve(),a.control.resolve(),a.code.resolve();condition=a.condition;reset=a.reset_index
if condition=='gptq_w4a16':import pi05_gptq_catalog_20261116_v1
auth=json.loads((c/'authorization.json').read_text());assert auth['hostname']==socket.gethostname() and auth['primary_only']
assert datetime.datetime.fromisoformat(auth['expires_at'])>datetime.datetime.now(datetime.timezone.utc)
parent=r/'rounds/20261111-084-pi05-smoothquant-a8-v2'
profile=(r/'control/20261116-084-pi05-resume-v1/stages/20261116-084-pi05-gptq-pilot-v4/profile/profile.pt') if condition=='gptq_w4a16' else parent/'profile/profile.pt'
bits=16 if condition=='gptq_w4a16' else 8;method='gptq' if bits==16 else 'smoothquant'
pr=json.loads(profile.with_name('result.json').read_text());assert pr['precision_gates'][str(bits)]=='PASS_RELOAD_FINITE' and digest(profile)==pr['profile_sha256']
backend='openpi-pytorch-original-independent-h10-v2';refs={};existing=set();active=[]
for stage in (c/'stages').iterdir():
    if (stage/'status.json').exists() and json.loads((stage/'status.json').read_text())['state'] in ['RUNNING','PREFLIGHT']:
        active.append(stage.resolve())
        conf=json.loads((stage/'config.json').read_text());series=json.loads((stage/'series.json').read_text())
        if conf['quantization']['method'] in ['none',method] and any(x['reset_index']==reset for x in series['cases']):raise ValueError('Requested reference or target cohort still active')
paths=[*(r/'rounds').glob('*/case-runs/*/result.json'),*(r/'control').glob('*/stages/*/case-runs/*/result.json')]
for f in paths:
    if not f.parent.name.endswith(f'-reset{reset}') or any(f.is_relative_to(x) for x in active):continue
    z=json.loads(f.read_text())
    if z.get('backend')!=backend or z.get('role')!='development' or z.get('reset_index')!=reset:continue
    key=(z['suite'],z['task_id'],reset);runtime=f.parent.parent.parent/'case-plans'/(z['round_id']+'-runtime.json')
    if z.get('condition')=='bf16':
        assert z['gate']=='PASS_PI05_TORCH_BF16_EPISODE' and key not in refs;audit_bf16(f.parent,runtime);refs[key]=(f,runtime)
    if z.get('condition')==condition:
        module='qvla.audit_pi05_torch_gptq_trace_v1' if bits==16 else 'qvla.audit_pi05_torch_sq_trace_v2'
        importlib.import_module(module).audit(f.parent,runtime);assert key not in existing;existing.add(key)
suites=['libero_spatial','libero_object','libero_goal','libero_10'];expected={(s,t,reset) for s in suites for t in range(10)}
assert set(refs)==expected,'Forty same-backend same-initial references required'
missing=sorted(expected-existing);assert missing,'No missing cases; never repeat completed data'
rd=c/'stages'/a.round_id;rd.mkdir(exist_ok=False)
for d in ['case-plans','case-runs','case-logs','audits']:(rd/d).mkdir()
cases=[];inputs=[]
for suite,task,_ in missing:
    ref,runtime=refs[suite,task,reset];template=json.loads(runtime.read_text());name=f'{a.round_id}-{suite}-task{task}-reset{reset}'
    actor='qvla.execute_pi05_torch_gptq_episode_v1' if bits==16 else 'qvla.execute_pi05_torch_sq_episode_v2'
    template.update(round_id=name,server_id='084',hostname=auth['hostname'],gpu_uuid=auth['gpu_uuid'],runner=actor,condition=condition,role='development',suite=suite,task_id=task,reset_index=reset,model_seed=7,backend=backend,bf16_reference=str(ref.parent),output=str(rd/'case-runs'/name),registered_at=now())
    f=rd/'case-plans'/(name+'.json');write_json(f,template);cases.append(dict(round_id=name,suite=suite,task_id=task,reset_index=reset,role='development',template_sha256=digest(f)))
    inputs += [dict(path=str(f),sha256=digest(f)),dict(path=str(ref),sha256=digest(ref)),dict(path=str(ref.parent/'query-00000.npz'),sha256=digest(ref.parent/'query-00000.npz'))]
config=json.loads((parent/'config.json').read_text());config['task']=json.loads((code/'Ex_Reports/_tasks'/config['task']['id']/'task.json').read_text())
config['control'].update(start_disk_gib=8,refresh_confirmation_wait_seconds=120);config['control'].pop('weekly_close_remaining_percent',None)
config['quantization']=dict(method=method,weight_bits=4,activation_bits=bits,profiles=[dict(path=str(profile),sha256=digest(profile))])
for item in config['evaluation']['suites']:item.update(task_ids=list(range(10)),reset_offset=reset,resets_per_task=1)
b=config['binding']
for name in ['admission_argv','post_evaluation_argv','post_training_audit_argv','evaluation_audit_argv']:b.pop(name,None)
b.update(id='pi05-'+condition+'-missing-predeclared-v1',quantizers=[method],supports_task_subsets=True)
entry='run_pi05_gptq_cases_v1.py' if bits==16 else 'run_pi05_sq_cases_v2.py'
b['evaluation_argv']=['{python}','{repo}/scripts/trajectory_drift_20261111/'+entry,'--root',str(r),'--round-dir',str(rd),'--suite','{suite}','--role','development','--activation-bits',str(bits)]
b['gate_files']=[b['gate_files'][0],dict(path=str(profile.with_name('result.json')),sha256=digest(profile.with_name('result.json')),expected_gate=pr['gate'])]
write_json(rd/'config.json',config)
old=json.loads((parent/'series.json').read_text());write_json(rd/'series.json',dict(old,cases=cases,control=config['control'],profile=str(profile),scope='Only missing fixed five-initial cases; frozen model/profile/seeds/cadence/noise'))
plan=compile_plan(config,code,rd)
rels=['qvla/pi05_torch_backend_v2.py',actor.replace('.','/')+'.py','scripts/trajectory_drift_20261111/'+entry,'scripts/trajectory_drift_20261119/prepare_pi05_mature_cases_v1.py']
if bits==16:rels+=['pi05_gptq_catalog_20261116_v1.py']
for rel in rels:plan['source_sha256'][str(code/rel)]=digest(code/rel)
plan['inputs']+=inputs+[dict(path=str(rd/'series.json'),sha256=digest(rd/'series.json'))]
plan.pop('plan_sha256');plan['plan_sha256']=canonical(plan);verify_plan(plan);write_json(rd/'plan.json',plan)
write_json(rd/'CPU_PLAN_READY.json',dict(gate='PASS_MATURE_QUANT_FROZEN_PROFILE_AND_PAIRED_REFERENCES_CPU',plan_sha256=plan['plan_sha256'],cases=len(cases),completed_excluded=len(existing),gpu_not_started=True))
print(json.dumps(dict(stage=str(rd),plan_sha256=plan['plan_sha256'],cases=len(cases),condition=condition,gpu_not_started=True)))
