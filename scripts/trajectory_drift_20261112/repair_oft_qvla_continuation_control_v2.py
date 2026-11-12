"""Preserve rejected v1. External frozen sources are hashed inputs, not repo-relative snapshots."""
from pathlib import Path
import datetime,json,os,shutil,subprocess,sys
r=Path('/root/autodl-tmp/trajectory-drift-20261107');root=r/'control/20261112-061-oft-qvla-resume-v5';old=root/'stages/qvla-remaining62-v1';rd=root/'stages/qvla-remaining62-v2';code=r/'code-20261112-061-qvla-resume-v5'
assert json.loads((old/'status.json').read_text())['state']=='PAUSED_SERVER_LEFT_ON';assert not list((old/'case-runs').glob('*/result.json'))
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
sys.path.insert(0,str(code));os.chdir(code)
from qvla.infra.common import canonical,digest,write_json,now
from qvla.infra.plan import compile_plan,verify_plan
from qvla.infra.lifecycle import quota_decision
quota=json.loads((root/'quota.json').read_text());config=json.loads((old/'config.json').read_text());assert quota_decision(quota,config['control'])=='CONTINUE'
rd.mkdir();
for name in ['case-runs','case-logs','admission']:(rd/name).mkdir()
shutil.copyfile(old/'admission/result.json',rd/'admission/result.json')
config['binding']['evaluation_argv']=[str(rd) if x==str(old) else x for x in config['binding']['evaluation_argv']];config['binding']['evaluation_audit_argv']=[str(rd) if x==str(old) else str(rd/'paired-audit') if x==str(old/'paired-audit') else x for x in config['binding']['evaluation_audit_argv']]
config['binding']['id']='qvla-same80-only62-external-source-snapshot-v2'
series=json.loads((old/'series.json').read_text())
for case in series['cases']:case['id']=case['id'].replace(old.name,rd.name)
write_json(rd/'config.json',config);write_json(rd/'series.json',series);write_json(rd/'quota.json',quota)
plan=compile_plan(config,code,rd);old_plan=json.loads((old/'plan.json').read_text())
for path,sha in old_plan['source_sha256'].items():
 f=Path(path);assert digest(f)==sha
 if f.is_relative_to(code):plan['source_sha256'][path]=sha
 else:plan['inputs'].append(dict(path=path,sha256=sha,role='external immutable scientific source; no repo-relative snapshot'))
for f in [rd/'series.json',rd/'admission/result.json',Path(series['gates'])]:plan['inputs'].append(dict(path=str(f),sha256=digest(f)))
plan.pop('plan_sha256');plan['plan_sha256']=canonical(plan);verify_plan(plan);write_json(rd/'plan.json',plan)
auth=json.loads((old/'authorization.json').read_text());auth.update(round_id=rd.name,config_sha256=canonical(config));write_json(rd/'authorization.json',auth)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',PYTHONPATH=str(code),MUJOCO_GL='egl',TOKENIZERS_PARALLELISM='false')
argv=[sys.executable,'-u','-m','qvla.infra','run',str(rd/'plan.json'),'--authorization',str(rd/'authorization.json'),'--round-id',rd.name,'--quota',str(rd/'quota.json'),'--execute']
with (rd/'supervisor.log').open('xb') as f:p=subprocess.Popen(argv,cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
rec=dict(supervisor_pid=p.pid,stage=str(rd),plan_sha256=plan['plan_sha256'],started_at=now(),new_cases=62,failed_v1_preserved=True,scientific_sources_unchanged=True);write_json(rd/'launch-receipt.json',rec)
current=json.loads((root/'CURRENT_RUN.json').read_text());current['stage']=str(rd);write_json(root/'CURRENT_RUN.json',current);shutil.copyfile(Path(__file__),root/'control-code-preserved'/Path(__file__).name);print(json.dumps(rec),flush=True)
