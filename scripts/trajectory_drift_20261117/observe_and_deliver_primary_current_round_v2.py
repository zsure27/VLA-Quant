from pathlib import Path
import json,os,subprocess,datetime,shutil,base64,sys
r=Path('/root/autodl-tmp/trajectory-drift-20261107');root=r/'control/20261117-084-pi05-resume-v1'
sys.path.insert(0,str(root))
q=json.loads(base64.b64decode(os.environ['VLA_QUOTA_B64']))
weekly_close=False
def atomic(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value));os.replace(tmp,path)
atomic(r/'084-primary-current-20261117-v1.json',q)
atomic(root/'quota.json',q)
for f in (root/'stages').glob('*/quota.json'):atomic(f,q)

guard=json.loads((root/'guard-status.json').read_text())
out=dict(observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),quota=q,guard=guard,
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),disk_free_bytes=shutil.disk_usage(r).free)
stage=root/'CURRENT_RUN.json'
if stage.exists():
 rd=Path(json.loads(stage.read_text())['stage']);s=json.loads((rd/'status.json').read_text())
 out.update(state=s['state'],stage=s.get('current_stage'),reason=s.get('reason'),stages=[dict(id=x['id'],exit=x.get('exit_code')) for x in s['stages']])
 if s['state'] not in ['RUNNING','PREFLIGHT']:out['terminal_log']=(rd/'supervisor.log').read_text()[-3000:]
out['processes']=[dict(pid=int(parts[0]),ppid=int(parts[1]),cpu_percent=float(parts[2]),rss_kib=int(parts[3]),program=parts[4].split()[0]) for line in subprocess.check_output(['ps','-eo','pid,ppid,pcpu,rss,args','--no-headers'],text=True).splitlines() if str(root) in line and (parts:=line.split(None,4))]
if guard['state'] not in ['CLOSING','SHUTDOWN_BLOCKED_GPU_NOT_IDLE'] and not weekly_close:
 atomic(root/'status.json',dict(state=out.get('state','PREFLIGHT'),active_stage_control=str(rd) if stage.exists() else None,observed_at=out['observed_at'],current_round_only=True))
 atomic(root/'heartbeat-latest.json',out)
print(json.dumps(out))
