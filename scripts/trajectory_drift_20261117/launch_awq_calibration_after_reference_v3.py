from pathlib import Path
import os,json,sys,datetime,socket,shutil,subprocess
r=Path('/root/autodl-tmp/trajectory-drift-20261107');root=r/'control/20261117-084-pi05-resume-v1';rd=root/'stages/20261117-084-pi05-awq-calibration-v3';code=r/'code-084-native-runtime-20261117-v2'
sys.path[:0]=[str(code),str(root)]
from qvla.infra.common import digest,canonical,write_json,now
from qvla.infra.plan import verify_plan
from guard_quota_window_v1 import bounded_quota_decision
q=json.loads((root/'quota.json').read_text());assert 'weekly' not in q
assert bounded_quota_decision(q,dict(quota_max_age_seconds=1020,close_remaining_percent=10,stop_expansion_remaining_percent=15))=='CONTINUE'
assert not (root/'CLOSE_NOW').exists() and not (root/'STOP_NEW_GPU').exists()
auth=json.loads((root/'authorization.json').read_text());assert auth['hostname']==socket.gethostname()
assert auth['gpu_uuid']==subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert shutil.disk_usage(r).free>=12*1024**3 and json.loads((root/'guard-status.json').read_text())['state']=='CONTINUATION_ARMED'
current=json.loads((root/'CURRENT_RUN.json').read_text());assert current['condition']=='bf16' and 'initial35-v3' in current['stage']
previous=Path(current['stage']);state=json.loads((previous/'status.json').read_text());assert state['state']=='PAUSED_SERVER_LEFT_ON' and len(state['stages'])==4 and all(x['exit_code']==0 for x in state['stages'])
analysis=Path(os.environ['VLA_PREDECESSOR_ANALYSIS']);az=json.loads(analysis.read_text())
assert az['gate']=='PASS_AVAILABLE_PI05_SAME_BACKEND_RAW_GEOMETRY_AUDIT' and az['reset_indices']==[35]
assert az['coverage']['bf16']['complete']==40
prior25=root/'analysis-initial25-complete-v1/summary-public.json';zz=json.loads(prior25.read_text());assert all(zz['coverage'][c]['complete']==40 for c in ['bf16','smoothquant_w4a8','gptq_w4a16'])
plan=json.loads((rd/'plan.json').read_text());ready=json.loads((rd/'CPU_PLAN_READY.json').read_text());assert ready['plan_sha256']==plan['plan_sha256']
verify_plan(plan);assert plan['execution_scope']=='CALIBRATION_ONLY_ZERO_FORMAL_EPISODES' and len(plan['stages'])==1
assert not (rd/'launch-receipt.json').exists() and not (rd/'status.json').exists()
execution=dict(plan,parent_cpu_plan_sha256=plan['plan_sha256']);execution['inputs']=[*plan['inputs'],dict(path=str(analysis),sha256=digest(analysis)),dict(path=str(prior25),sha256=digest(prior25)),dict(path=str(previous/'status.json'),sha256=digest(previous/'status.json'))]
execution.pop('plan_sha256');execution['plan_sha256']=canonical(execution);verify_plan(execution);write_json(rd/'execution-plan-v1.json',execution)
authorization=dict(auth,round_id=rd.name,shutdown_authorized=False,pause_only=True,config_sha256=execution['config_sha256'],control_code=str(code),expires_at=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=8)).isoformat(),source='Current user continuation; predeclared original AWQ calibration after GPTQ audit; zero formal episodes; Windows stays on')
write_json(rd/'authorization.json',authorization);write_json(rd/'quota.json',q)
args=[sys.executable,'-u','-m','qvla.infra','run',str(rd/'execution-plan-v1.json'),'--authorization',str(rd/'authorization.json'),'--round-id',rd.name,'--quota',str(rd/'quota.json'),'--execute']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='4',PYTHONPATH=':'.join([str(code),str(r/'sources/openpi/src'),str(r/'sources/openpi/packages/openpi-client/src'),str(r/'sources/LIBERO')]))
with (rd/'supervisor.log').open('xb') as f:p=subprocess.Popen(args,cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
write_json(rd/'launch-receipt.json',dict(supervisor_pid=p.pid,plan_sha256=execution['plan_sha256'],parent_cpu_plan_sha256=plan['plan_sha256'],started_at=now(),formal_episodes=0))
write_json(root/'CURRENT_RUN.json',dict(stage=str(rd),condition='awq_calibration_only',control_code=str(code)))
print('AWQ_CALIBRATION_ONLY_LAUNCHED',p.pid,execution['plan_sha256'])
