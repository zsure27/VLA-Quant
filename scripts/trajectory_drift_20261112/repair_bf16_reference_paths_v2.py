"""Preserve rejected no-model v1; supply existing OFT/LIBERO roots through infra environment."""
from pathlib import Path
import json,os,subprocess,sys
r=Path('/root/autodl-tmp/trajectory-drift-20261107');root=r/'control/20261112-061-oft-qvla-resume-v6';old=root/'stages/bf16-initial25-v1';rd=root/'stages/bf16-initial25-v2';code=r/'code-20261112-061-qvla-resume-v5';assert json.loads((old/'status.json').read_text())['state']=='PAUSED_SERVER_LEFT_ON' and not list((old/'case-runs').glob('*/result.json'))
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();sys.path.insert(0,str(code));os.chdir(code)
from qvla.infra.common import canonical,digest,write_json,now
from qvla.infra.plan import compile_plan,verify_plan
from qvla.infra.lifecycle import quota_decision
config=json.loads((old/'config.json').read_text());quota=json.loads((root/'quota.json').read_text());assert quota_decision(quota,config['control'])=='CONTINUE'
config['model'].update(oft_root=str(r/'sources/openvla-oft'),libero_root=str(r/'sources/LIBERO'));config['binding']['id']='native-bf16-initial25-explicit-source-roots-v2';config['binding']['evaluation_argv']=[str(rd) if x==str(old) else x for x in config['binding']['evaluation_argv']];config['binding']['evaluation_audit_argv']=[str(rd) if x==str(old) else x for x in config['binding']['evaluation_audit_argv']]
rd.mkdir();(rd/'case-runs').mkdir();(rd/'case-logs').mkdir();series=json.loads((old/'series.json').read_text())
for c in series['cases']:c['id']=c['id'].replace(old.name,rd.name)
write_json(rd/'series.json',series);write_json(rd/'config.json',config);write_json(rd/'quota.json',quota);plan=compile_plan(config,code,rd)
for row in json.loads((old/'plan.json').read_text())['inputs']:
 if row['path']==str(old/'series.json'):continue
 assert digest(Path(row['path']))==row['sha256'];plan['inputs'].append(row)
plan['inputs'].append(dict(path=str(rd/'series.json'),sha256=digest(rd/'series.json')));plan.pop('plan_sha256');plan['plan_sha256']=canonical(plan);verify_plan(plan);write_json(rd/'plan.json',plan)
auth=json.loads((old/'authorization.json').read_text());auth.update(round_id=rd.name,config_sha256=canonical(config));write_json(rd/'authorization.json',auth)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',PYTHONPATH=str(code),MUJOCO_GL='egl',TOKENIZERS_PARALLELISM='false');argv=[sys.executable,'-u','-m','qvla.infra','run',str(rd/'plan.json'),'--authorization',str(rd/'authorization.json'),'--round-id',rd.name,'--quota',str(rd/'quota.json'),'--execute']
with (rd/'supervisor.log').open('xb') as f:p=subprocess.Popen(argv,cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
rec=dict(supervisor_pid=p.pid,stage=str(rd),plan_sha256=plan['plan_sha256'],started_at=now(),new_reference_cases=40,old_first_case_import_failure_preserved=True,scientific_sources_unchanged=True);write_json(rd/'launch-receipt.json',rec);current=json.loads((root/'CURRENT_RUN.json').read_text());current['stage']=str(rd);write_json(root/'CURRENT_RUN.json',current);print(json.dumps(rec),flush=True)
