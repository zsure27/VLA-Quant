"""CPU-only registration of missing predeclared BF16 references; never launches."""
import argparse,datetime,json,socket
import pi05_quant_catalog_20261118_v1  # Verify the frozen teacher-admission plan.
from pathlib import Path
from qvla.infra.common import digest,canonical,write_json,now
from qvla.infra.plan import compile_plan,verify_plan
from qvla.audit_pi05_torch_trace_v2 import audit

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--control',type=Path,required=True);p.add_argument('--code',type=Path,required=True);p.add_argument('--reset-index',type=int,choices=[35,45],required=True);p.add_argument('--round-id',required=True);p.add_argument('--teacher-stage',type=Path,required=True);a=p.parse_args()
r,c,code=a.root.resolve(),a.control.resolve(),a.code.resolve();parent=r/'rounds/20261111-084-pi05-torch-independent-v2'
auth=json.loads((c/'authorization.json').read_text());assert auth['hostname']==socket.gethostname() and auth['primary_only']
assert datetime.datetime.fromisoformat(auth['expires_at'])>datetime.datetime.now(datetime.timezone.utc)
previous=json.loads((parent/'plan.json').read_text());verify_plan(previous)
teacher_plan=json.loads((a.teacher_stage/'plan.json').read_text());verify_plan(teacher_plan)
teacher_gate=a.teacher_stage/'profile/result.json';teacher=json.loads(teacher_gate.read_text())
assert teacher['gate']=='PASS_PI05_OMNI_LWC_NATIVE_414_RELOAD_FINITE' and teacher['teacher_actions_restored_bitwise'] is True
backend='openpi-pytorch-original-independent-h10-v2';assert teacher['backend']==backend
existing={};paths=[*(r/'rounds').glob('*/case-runs/*/result.json'),*(r/'control').glob('*/stages/*/case-runs/*/result.json')]
for f in paths:
    stage=f.parent.parent.parent
    # Only inspect immutable BF16 stages or the original independent-BF16 round.
    if stage!=parent and 'bf16' not in stage.name:continue
    if (stage/'status.json').exists():
        state=json.loads((stage/'status.json').read_text())
        if state['state'] in ['RUNNING','PREFLIGHT']:
            # An old RUNNING label does not establish a live process after restart.
            receipt=stage/'launch-receipt.json'
            launch=json.loads(receipt.read_text()) if receipt.exists() else {}
            for pid in [state.get('child_pid',0),launch.get('supervisor_pid',0)]:
                proc=Path('/proc',str(pid),'cmdline')
                if proc.exists() and str(stage).encode() in proc.read_bytes():
                    raise ValueError('BF16 stage actually active; immutable audit required')
    z=json.loads(f.read_text())
    if z.get('backend')!=backend or z.get('condition')!='bf16' or z.get('role')!='development' or z.get('reset_index')!=a.reset_index:continue
    assert z['gate']=='PASS_PI05_TORCH_BF16_EPISODE';key=(z['suite'],z['task_id'],z['reset_index']);assert key not in existing
    runtime=stage/'case-plans'/(z['round_id']+'-runtime.json');proof=audit(f.parent,runtime)
    existing[key]=dict(result_path=str(f),result_sha256=digest(f),audit_gate=proof['gate'])
old=json.loads((parent/'series.json').read_text());prototypes=[x for x in old['cases'] if x['role']=='development'];assert len(prototypes)==4
rd=c/'stages'/a.round_id;rd.mkdir(exist_ok=False)
for name in ['case-plans','case-runs','case-logs','audits','admission']:(rd/name).mkdir()
admission=parent/'admission/result.json';(rd/'admission/result.json').write_bytes(admission.read_bytes())
cases=[];inputs=[]
for source in prototypes:
    template=json.loads((parent/'case-plans'/(source['round_id']+'.json')).read_text())
    for task in range(10):
        key=(source['suite'],task,a.reset_index)
        if key in existing:continue
        name=f'{a.round_id}-{source["suite"]}-task{task}-reset{a.reset_index}'
        item=dict(template,round_id=name,task_id=task,reset_index=a.reset_index,hostname=auth['hostname'],gpu_uuid=auth['gpu_uuid'],server_id='084',output=str(rd/'case-runs'/name),registered_at=now())
        f=rd/'case-plans'/(name+'.json');write_json(f,item);inputs.append(dict(path=str(f),sha256=digest(f)))
        cases.append(dict(source,task_id=task,reset_index=a.reset_index,round_id=name,template_sha256=digest(f)))
assert cases and len(cases)+len(existing)==40
config=json.loads((parent/'config.json').read_text());config['task']=json.loads((code/'Ex_Reports/_tasks'/config['task']['id']/'task.json').read_text())
assert config['model']['identity_sha256']==teacher['model_identity']
config['control'].update(start_disk_gib=8,refresh_confirmation_wait_seconds=120);config['control'].pop('weekly_close_remaining_percent',None)
for suite in config['evaluation']['suites']:suite.update(task_ids=list(range(10)),reset_offset=a.reset_index,resets_per_task=1)
b=config['binding'];b.pop('admission_argv',None);b.pop('post_evaluation_argv',None)
b['evaluation_argv']=['{python}','{repo}/scripts/trajectory_drift_20261111/run_pi05_torch_cases_v2.py','--root',str(r),'--round-dir',str(rd),'--suite','{suite}','--role','development']
b['gate_files'] += [dict(path=str(rd/'admission/result.json'),sha256=digest(admission),expected_gate='PASS_PI05_TORCH_INDEPENDENT_BF16_RELOAD_INPUT'),dict(path=str(teacher_gate),sha256=digest(teacher_gate),expected_gate=teacher['gate'])]
write_json(rd/'config.json',config);write_json(rd/'series.json',dict(old,cases=cases,control=config['control'],scope='Only missing predeclared BF16 cases; original 10x32/execute5, model7/env0/noise0 unchanged'))
write_json(rd/'COMPLETED_BF16_EXCLUSION.json',dict(reset_index=a.reset_index,completed=list(existing.values()),new_cases=len(cases),original_interrupted_stage_not_rewritten=True))
plan=compile_plan(config,code,rd)
for rel in ['pi05_quant_catalog_20261118_v1.py','qvla/pi05_torch_backend_v2.py','qvla/execute_pi05_torch_episode_v2.py','qvla/audit_pi05_torch_trace_v2.py','scripts/trajectory_drift_20261111/run_pi05_torch_cases_v2.py','scripts/trajectory_drift_20261118/prepare_pi05_bf16_missing_v1.py']:plan['source_sha256'][str(code/rel)]=digest(code/rel)
plan['inputs']+=inputs+[dict(path=str(rd/'series.json'),sha256=digest(rd/'series.json')),dict(path=str(rd/'COMPLETED_BF16_EXCLUSION.json'),sha256=digest(rd/'COMPLETED_BF16_EXCLUSION.json'))]
plan.pop('plan_sha256');plan['plan_sha256']=canonical(plan);verify_plan(plan);write_json(rd/'plan.json',plan)
write_json(rd/'CPU_PLAN_READY.json',dict(gate='PASS_MISSING_PREDECLARED_BF16_REFERENCES_CPU_PLAN',plan_sha256=plan['plan_sha256'],completed_excluded=len(existing),cases=len(cases),gpu_not_started=True))
print(json.dumps(dict(stage=str(rd),plan_sha256=plan['plan_sha256'],cases=len(cases),completed_excluded=len(existing),gpu_not_started=True)))
