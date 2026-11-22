"""Explicit single-stage controller action; never schedules a following stage."""
import argparse,datetime,importlib,json,os,shutil,socket,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--control',type=Path,required=True);p.add_argument('--code',type=Path,required=True);p.add_argument('--stage',type=Path,required=True);p.add_argument('--plugin');p.add_argument('--minimum-disk-gib',type=int,default=12);a=p.parse_args()
r,control,code,stage=a.root.resolve(),a.control.resolve(),a.code.resolve(),a.stage.resolve()
assert stage.parent==control/'stages' and a.minimum_disk_gib>=8
sys.path[:0]=[str(control),str(code)]
if a.plugin:importlib.import_module(a.plugin)
from qvla.infra.common import canonical,write_json,now
from qvla.infra.plan import verify_plan
from guard_quota_window_v1 import bounded_quota_decision
plan=json.loads((stage/'plan.json').read_text());auth=json.loads((control/'authorization.json').read_text());quota=json.loads((control/'quota.json').read_text())
assert not (stage/'status.json').exists() and not (stage/'launch-receipt.json').exists()
assert not (control/'CLOSE_NOW').exists() and not (control/'STOP_NEW_GPU').exists()
assert json.loads((control/'guard-status.json').read_text())['state']=='CONTINUATION_ARMED'
assert bounded_quota_decision(quota,plan['config']['control'])=='CONTINUE' and 'weekly' not in quota
assert datetime.datetime.fromisoformat(auth['expires_at'])>datetime.datetime.now(datetime.timezone.utc)
assert auth['hostname']==socket.gethostname() and auth['gpu_uuid']==subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert shutil.disk_usage(r).free>=a.minimum_disk_gib*1024**3
previous=Path(json.loads((control/'CURRENT_RUN.json').read_text())['stage']);state=json.loads((previous/'status.json').read_text())
assert state['state']=='PAUSED_SERVER_LEFT_ON' and all(x.get('exit_code')==0 for x in state['stages'])
verify_plan(plan);assert json.loads((stage/'CPU_PLAN_READY.json').read_text())['plan_sha256']==plan['plan_sha256']
write_json(stage/'quota.json',quota);write_json(stage/'authorization.json',dict(auth,round_id=stage.name,config_sha256=canonical(plan['config']),shutdown_authorized=False,pause_only=True))
args=[sys.executable,'-u','-m','qvla.infra']+(['--plugin',a.plugin] if a.plugin else [])+['run',str(stage/'plan.json'),'--authorization',str(stage/'authorization.json'),'--round-id',stage.name,'--quota',str(stage/'quota.json'),'--execute']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='4',PYTHONPATH=':'.join([str(code),str(r/'sources/openpi/src'),str(r/'sources/openpi/packages/openpi-client/src'),str(r/'sources/LIBERO')]))
with (stage/'supervisor.log').open('xb') as log:child=subprocess.Popen(args,cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
write_json(stage/'launch-receipt.json',dict(supervisor_pid=child.pid,argv=args,started_at=now(),plan_sha256=plan['plan_sha256']))
write_json(control/'CURRENT_RUN.json',dict(stage=str(stage),condition=plan['config']['quantization']['method'],control_code=str(code)))
print(json.dumps(dict(state='PREPARED_SINGLE_STAGE_LAUNCHED',stage=str(stage),pid=child.pid,plan_sha256=plan['plan_sha256'])))
