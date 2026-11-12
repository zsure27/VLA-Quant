"""Independently locked continuation of9 missing cases after a preserved control interruption."""
from pathlib import Path
import json,os,subprocess,sys,shutil,datetime,socket
r=Path('/root/autodl-tmp/trajectory-drift-20261107');oldroot=r/'control/20261112-061-oft-qvla-resume-v5';old=oldroot/'stages/qvla-remaining62-v2';root=r/'control/20261112-061-oft-qvla-resume-v6';rd=root/'stages/qvla-remaining9-v1';code=r/'code-20261112-061-qvla-resume-v5';parent=r/'control/20261112-061-oft-other-configs-v4'
sys.path.insert(0,str(code));os.chdir(code)
from qvla.infra.common import canonical,digest,write_json,now
from qvla.infra.plan import compile_plan,verify_plan
from qvla.infra.lifecycle import quota_decision
config=json.loads((old/'config.json').read_text());quota=json.loads((root/'quota.json').read_text());assert quota_decision(quota,config['control'])=='CONTINUE'
auditfile=old/'control-interruption-audit/result.json';audit=json.loads(auditfile.read_text());assert audit['gate']=='PASS_QVLA_AVAILABLE_RAW_PAIRS_CONTROL_INTERRUPTION' and audit['completed']==53
assert json.loads((old/'status.json').read_text())['state']=='PAUSED_SERVER_LEFT_ON' and not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
authroot=json.loads((root/'authorization.json').read_text());assert socket.gethostname()==authroot['hostname'] and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==authroot['gpu_uuid'];os.kill(json.loads((root/'guard-launch-receipt.json').read_text())['pid'],0)
done={};parents=[dict(stage=str(parent/'stages/qvla-pilot-v4'),audit='paired-audit/result.json'),dict(stage=str(parent/'stages/qvla-complete80-v1'),audit='paired-audit/result.json'),dict(stage=str(old),audit='control-interruption-audit/result.json')]
for rec in parents:
 stage=Path(rec['stage']);proof=json.loads((stage/rec['audit']).read_text());assert proof['gates_sha256']==audit['gates_sha256']
 for f in stage.glob('libero*-cases.json'):
  for row in json.loads(f.read_text())['completed']:
   case=row['case'];key=(case['suite'],case['task_id'],case['reset_index']);assert key not in done and digest(Path(row['result_path']))==row['result_sha256'];done[key]=row
assert len(done)==71
# Orphan full result files are not silently admitted; require ledger and post-query proofs.
found={}
for p in (r/'control').glob('*/stages/*/case-runs/*/result.json'):
 z=json.loads(p.read_text())
 if z.get('condition')=='qvla_w4a16' and z.get('model')=='openvla_oft':
  key=(z['suite'],z['task_id'],z['initial_state_index']);assert key not in found;found[key]=str(p)
assert set(found)==set(done),'Uncatalogued result requires separate segmented audit before any rerun'
oldseries=json.loads((old/'series.json').read_text());cases=[]
for c in oldseries['cases']:
 if (c['suite'],c['task_id'],c['reset_index']) in done:continue
 cases.append(dict(c,id=rd.name+'-'+c['suite']+f"-task{c['task_id']}-reset{c['reset_index']}"))
assert len(cases)==9 and all(c['suite']=='libero_10' and c['reset_index']==15 for c in cases)
rd.mkdir()
for name in ['case-runs','case-logs','admission']:(rd/name).mkdir()
shutil.copyfile(old/'admission/result.json',rd/'admission/result.json');config['binding']['id']='qvla-same80-nine-missing-after-control-interruption-v1'
config['binding']['evaluation_argv']=[str(rd) if x==str(old) else x for x in config['binding']['evaluation_argv']];config['binding']['evaluation_audit_argv']=[str(rd) if x==str(old) else str(rd/'paired-audit') if x==str(old/'paired-audit') else x for x in config['binding']['evaluation_audit_argv']]
config['binding']['gate_files'].append(dict(path=str(auditfile),sha256=digest(auditfile),expected_gate=audit['gate']));config['scope'].update(actual_stage='9 missing Long reset15/task1-9; prior71 paired audited; preserved interrupted attempt is not failure or denominator',prior_control=str(oldroot),current_guard_root=str(root))
series=dict(oldseries,cases=cases,root_control=str(root),prior_complete_qvla_cases=71,prior_qvla_stages=parents,expected_total_qvla_cases=80,control_interruption_preserved=True)
write_json(rd/'series.json',series);write_json(rd/'config.json',config);write_json(rd/'quota.json',quota)
plan=compile_plan(config,code,rd)
for path,sha in json.loads((old/'plan.json').read_text())['source_sha256'].items():assert digest(Path(path))==sha;plan['source_sha256'][path]=sha
for row in json.loads((old/'plan.json').read_text())['inputs']:
 if row['path'] in [str(old/'series.json'),str(old/'admission/result.json')]:continue
 assert digest(Path(row['path']))==row['sha256'];plan['inputs'].append(row)
for f in [rd/'series.json',rd/'admission/result.json']:plan['inputs'].append(dict(path=str(f),sha256=digest(f)))
plan.pop('plan_sha256');plan['plan_sha256']=canonical(plan);verify_plan(plan);write_json(rd/'plan.json',plan)
auth=json.loads((old/'authorization.json').read_text());auth.update(round_id=rd.name,config_sha256=canonical(config),source='Same current user-opened061, actual58percent; independent missing-case continuation after preserved control interruption');write_json(rd/'authorization.json',auth)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',PYTHONPATH=str(code),MUJOCO_GL='egl',TOKENIZERS_PARALLELISM='false');argv=[sys.executable,'-u','-m','qvla.infra','run',str(rd/'plan.json'),'--authorization',str(rd/'authorization.json'),'--round-id',rd.name,'--quota',str(rd/'quota.json'),'--execute']
with (rd/'supervisor.log').open('xb') as f:p=subprocess.Popen(argv,cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
rec=dict(supervisor_pid=p.pid,stage=str(rd),plan_sha256=plan['plan_sha256'],started_at=now(),new_cases=9,prior_complete_qvla=71,preserved_control_interruption=True);write_json(rd/'launch-receipt.json',rec);write_json(root/'CURRENT_RUN.json',dict(control_code=str(code),stage=str(rd),gpu_uuid=authroot['gpu_uuid']));write_json(root/'status.json',dict(state='RUNNING',round_id=root.name,started_at=now(),parent_control=str(oldroot)))
shutil.copytree(oldroot/'cpu-preparation/bf16-initial25-blueprint-v1',root/'cpu-preparation/bf16-initial25-blueprint-v1');saved=root/'control-code-preserved';saved.mkdir();shutil.copyfile(Path(__file__),saved/Path(__file__).name);print(json.dumps(rec),flush=True)
