"""Current-host continuation guard: stage success pauses; actual quota closes.

No policy runs here. Watches the current root quota, stops only registered own
infra process groups, then invokes the existing validated native provider.
"""
import argparse,datetime,json,os,pathlib,signal,socket,subprocess,sys,time
p=argparse.ArgumentParser();p.add_argument('--root-control',type=pathlib.Path,required=True);p.add_argument('--relay-seconds',type=int,default=600);a=p.parse_args();assert 180<=a.relay_seconds<=900;rd=a.root_control
auth=json.loads((rd/'authorization.json').read_text());assert auth['shutdown_authorized'] and auth['hostname']==socket.gethostname()
from qvla.infra.common import write_json,now
from qvla.infra.lifecycle import native_shutdown
from guard_quota_window_v1 import bounded_quota_decision as quota_decision
control=dict(quota_max_age_seconds=1020,close_remaining_percent=10,stop_expansion_remaining_percent=15,refresh_confirmation_wait_seconds=120)
paused_groups=set()
def pause_own_groups():
 if paused_groups:return
 rows=[x.split(None,2) for x in subprocess.check_output(['ps','-eo','pid,ppid,args','--no-headers'],text=True).splitlines()]
 owned=set()
 for f in (rd/'stages').glob('*/launch-receipt.json'):
  pid=json.loads(f.read_text())['supervisor_pid'];proc=pathlib.Path('/proc',str(pid),'cmdline')
  if proc.exists() and str(f.parent).encode() in proc.read_bytes():owned.add(pid)
 changed=True
 while changed:
  changed=False
  for row in rows:
   if len(row)==3 and int(row[1]) in owned and int(row[0]) not in owned:owned.add(int(row[0]));changed=True
 for pid in owned:
  try:
   group=os.getpgid(pid)
   if group!=os.getpgrp():paused_groups.add(group)
  except ProcessLookupError:pass
 for group in paused_groups:
  try:os.killpg(group,signal.SIGSTOP)
  except ProcessLookupError:pass
 write_json(rd/'refresh-pause-receipt.json',dict(groups=sorted(paused_groups),time_utc=now(),no_new_quota_assumed=True))
def resume_own_groups():
 for group in paused_groups:
  try:os.killpg(group,signal.SIGCONT)
  except ProcessLookupError:pass
 if paused_groups:write_json(rd/'refresh-resume-receipt.json',dict(groups=sorted(paused_groups),time_utc=now()))
 paused_groups.clear()

while True:
 reason=None
 try:
  decision=quota_decision(json.loads((rd/'quota.json').read_text()),control)
  if decision=='WAIT_API_REFRESH':
   pause_own_groups()
   write_json(rd/'guard-status.json',dict(state='WAIT_API_REFRESH',time_utc=now(),bounded_wait_seconds=120))
   time.sleep(2)
   continue
  resume_own_groups()
  if decision=='CLOSE':reason='actual_primary_at_or_below_10'
  if decision=='STOP_EXPANSION':(rd/'STOP_NEW_GPU').touch()
 except Exception as e:reason='actual_primary_unavailable_or_expired: '+str(e)
 if (rd/'CLOSE_NOW').exists():reason='current_controller_requests_protect_and_close'
 if reason:
  resume_own_groups()
  write_json(rd/'guard-status.json',dict(state='CLOSING',reason=reason,time_utc=now()))
  for f in (rd/'stages').glob('*/launch-receipt.json'):
   row=json.loads(f.read_text());pid=row['supervisor_pid'];proc=pathlib.Path('/proc',str(pid),'cmdline')
   if proc.exists() and str(f.parent).encode() in proc.read_bytes():
    try:os.kill(pid,signal.SIGTERM)
    except ProcessLookupError:pass
  for _ in range(45):
   q=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True)
   if q.returncode==0 and not q.stdout.strip():break
   time.sleep(2)
  else:write_json(rd/'guard-status.json',dict(state='SHUTDOWN_BLOCKED_GPU_NOT_IDLE',time_utc=now()));raise RuntimeError('GPU not idle; no native request')
  # Preserve originals even if controller/cloud access ends. No credential export.
  write_json(rd/'guard-close-record.json',dict(reason=reason,originals_retained=True,time_utc=now()))
  write_json(rd/'status.json',dict(state='INTERRUPTED',reason=reason,round_id=rd.name,ended_at=now()))
  code=pathlib.Path(auth['control_code'])
  with (rd/'backup.log').open('ab') as log:
   try:subprocess.run([sys.executable,str(code/'scripts/infra_archive_run.py'),'--run-dir',str(rd),'--prepare-only'],stdout=log,stderr=subprocess.STDOUT,timeout=300,check=True)
   except Exception as error:write_json(rd/'backup-prepare-error.json',dict(error=str(error),originals_retained=True))
  # Local controlling task owns private credentials. Bounded relay only; an
  # unsuccessful upload never blocks eventual native closure.
  deadline=time.monotonic()+a.relay_seconds
  while time.monotonic()<deadline:
   receipt=rd/'backup-receipt.json'
   if receipt.exists() and json.loads(receipt.read_text()).get('remote_download_sha256_verified'):break
   time.sleep(2)
  native_shutdown(rd,auth['hostname'],execute=True);break
 write_json(rd/'guard-status.json',dict(state='CONTINUATION_ARMED',time_utc=now(),shutdown_only_on_actual_quota_or_explicit_unrecoverable_close=True))
 time.sleep(60)  # Local quota/control files only, no per-episode/resource polling.
