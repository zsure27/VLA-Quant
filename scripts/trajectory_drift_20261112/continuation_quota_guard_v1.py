"""Current-host continuation guard: stage success pauses; actual quota closes.

No policy runs here. Watches the current root quota, stops only registered own
infra process groups, then invokes the existing validated native provider.
"""
import argparse,datetime,json,os,pathlib,signal,socket,subprocess,sys,time
p=argparse.ArgumentParser();p.add_argument('--root-control',type=pathlib.Path,required=True);a=p.parse_args();rd=a.root_control
auth=json.loads((rd/'authorization.json').read_text());assert auth['shutdown_authorized'] and auth['hostname']==socket.gethostname()
from qvla.infra.common import write_json,now
from qvla.infra.lifecycle import native_shutdown,quota_decision
control=dict(quota_max_age_seconds=1020,close_remaining_percent=10,stop_expansion_remaining_percent=15)
while True:
 reason=None
 try:
  decision=quota_decision(json.loads((rd/'quota.json').read_text()),control)
  if decision=='CLOSE':reason='actual_primary_at_or_below_10'
  if decision=='STOP_EXPANSION':(rd/'STOP_NEW_GPU').touch()
 except Exception as e:reason='actual_primary_unavailable_or_expired: '+str(e)
 if (rd/'CLOSE_NOW').exists():reason='current_controller_requests_protect_and_close'
 if reason:
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
   try:subprocess.run([sys.executable,str(code/'scripts/infra_archive_run.py'),'--run-dir',str(rd),'--prepare-only'],stdout=log,stderr=subprocess.STDOUT,timeout=120,check=True)
   except Exception as error:write_json(rd/'backup-prepare-error.json',dict(error=str(error),originals_retained=True))
  # Local controlling task owns private credentials. Bounded relay only; an
  # unsuccessful upload never blocks eventual native closure.
  deadline=time.monotonic()+180
  while time.monotonic()<deadline:
   receipt=rd/'backup-receipt.json'
   if receipt.exists() and json.loads(receipt.read_text()).get('remote_download_sha256_verified'):break
   time.sleep(2)
  native_shutdown(rd,auth['hostname'],execute=True);break
 write_json(rd/'guard-status.json',dict(state='CONTINUATION_ARMED',time_utc=now(),shutdown_only_on_actual_quota_or_explicit_unrecoverable_close=True))
 time.sleep(60)  # Local quota/control files only, no per-episode/resource polling.
