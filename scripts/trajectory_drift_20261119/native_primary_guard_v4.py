"""Quota, audited completion, or continuous 30-minute idle native closure. No experiment selection or server-side scheduler.

Missing/stale quota or expired operational authorization pauses owned model groups
and records attention, but is not mislabelled as a measured primary threshold.
"""
import argparse,datetime,json,os,pathlib,signal,socket,subprocess,sys,time
from primary_only_closure_policy_v1 import decision
from completion_idle_policy_v1 import completion_verified,idle_transition


def main():
    p=argparse.ArgumentParser();p.add_argument('--root-control',type=pathlib.Path,required=True)
    p.add_argument('--relay-seconds',type=int,default=600);a=p.parse_args()
    assert 180<=a.relay_seconds<=900
    rd=a.root_control.resolve();auth=json.loads((rd/'authorization.json').read_text())
    assert auth['hostname']==socket.gethostname() and auth['shutdown_authorized']
    assert auth['closure_policy']=='primary_or_audited_completion_or_gpu_idle_30min'
    assert auth.get('task_manifest_sha256')
    from qvla.infra.common import write_json,now
    from qvla.infra.lifecycle import native_shutdown
    paused=set();idle_since=None

    def owned_groups():
        rows=[x.split(None,2) for x in subprocess.check_output(['ps','-eo','pid,ppid,args','--no-headers'],text=True).splitlines()]
        owned=set()
        for f in (rd/'stages').glob('*/launch-receipt.json'):
            pid=json.loads(f.read_text())['supervisor_pid'];proc=pathlib.Path('/proc',str(pid),'cmdline')
            if proc.exists() and str(f.parent).encode() in proc.read_bytes():owned.add(pid)
        changed=True
        while changed:
            changed=False
            for row in rows:
                if len(row)==3 and int(row[1]) in owned and int(row[0]) not in owned:
                    owned.add(int(row[0]));changed=True
        result=set()
        for pid in owned:
            try:
                group=os.getpgid(pid)
                if group!=os.getpgrp():result.add(group)
            except ProcessLookupError:pass
        return result

    def send(groups,sig):
        for group in groups:
            try:os.killpg(group,sig)
            except ProcessLookupError:pass

    while True:
        try:quota=json.loads((rd/'quota.json').read_text())
        except (OSError,ValueError):quota=None
        try:
            expires=datetime.datetime.fromisoformat(auth['expires_at'].replace('Z','+00:00'))
            valid=datetime.datetime.now(datetime.timezone.utc)<expires
        except (KeyError,ValueError):valid=False
        mode=decision(quota,authorization_valid=valid)
        try:completion=json.loads((rd/'TASK_COMPLETION_AUDIT.json').read_text())
        except (OSError,ValueError):completion=None
        if completion_verified(completion,auth,rd):
            mode='CLOSE_AUDITED_TASK_COMPLETE'
        else:
            try:
                devices=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15).strip().splitlines()
                assert len(devices)==1
                uuid,memory,util=[x.strip() for x in devices[0].split(',')]
                assert uuid==auth['gpu_uuid']
                compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=15).strip()
                idle_since,idle_close=idle_transition(idle_since,time.monotonic(),gpu_present=True,observation_valid=True,
                    compute_processes_present=bool(compute),owned_work_present=bool(owned_groups()),
                    memory_used_mib=int(memory),utilization_percent=int(util))
            except Exception:
                idle_since=None;idle_close=False
            write_json(rd/'gpu-idle-watch.json',dict(observed_at=now(),idle_since_monotonic=idle_since,
                threshold_seconds=1800,idle_close=idle_close,no_experiment_scheduling=True))
            if idle_close:mode='CLOSE_GPU_IDLE_30_MINUTES'
        if mode.startswith('PAUSE_'):
            groups=owned_groups();send(groups,signal.SIGSTOP);paused.update(groups)
            (rd/'STOP_NEW_GPU').touch()
            write_json(rd/'guard-status.json',dict(state='WAIT_API_REFRESH',
                reason=mode,time_utc=now(),native_shutdown_requested=False,
                attention_required=True,originals_retained=True))
        elif mode.startswith('CLOSE_'):
            write_json(rd/'guard-status.json',dict(state='CLOSING',reason=mode,time_utc=now()))
            send(paused,signal.SIGCONT);send(owned_groups(),signal.SIGTERM)
            for _ in range(45):
                gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
                if gpu.returncode==0 and not gpu.stdout.strip():break
                time.sleep(2)
            else:
                write_json(rd/'guard-status.json',dict(state='SHUTDOWN_BLOCKED_GPU_NOT_IDLE',time_utc=now()))
                raise RuntimeError('GPU not idle; no native stop request')
            write_json(rd/'guard-close-record.json',dict(reason=mode,quota_observed_at=(quota or {}).get('observed_at'),
                remaining_percent=(quota or {}).get('remaining_percent'),time_utc=now(),originals_retained=True))
            write_json(rd/'status.json',dict(state='COMPLETED' if mode=='CLOSE_AUDITED_TASK_COMPLETE' else 'INTERRUPTED',reason=mode,round_id=rd.name,ended_at=now()))
            with (rd/'backup.log').open('ab') as log:
                try:subprocess.run([sys.executable,str(pathlib.Path(auth['control_code'])/'scripts/infra_archive_run.py'),
                    '--run-dir',str(rd),'--prepare-only'],stdout=log,stderr=subprocess.STDOUT,timeout=300,check=True)
                except Exception as error:write_json(rd/'backup-prepare-error.json',dict(error_type=type(error).__name__,originals_retained=True))
            until=time.monotonic()+a.relay_seconds
            while time.monotonic()<until:
                receipt=rd/'backup-receipt.json'
                if receipt.exists() and json.loads(receipt.read_text()).get('remote_download_sha256_verified'):break
                time.sleep(2)
            native_shutdown(rd,auth['hostname'],execute=True)
            return
        else:
            # Recovery is explicit. A paused group is not silently resumed after
            # a lease failure or transient outage; controller must audit/resume.
            if paused:
                write_json(rd/'guard-status.json',dict(state='WAIT_API_REFRESH',reason='FRESH_PRIMARY_RECOVERED_AWAITING_CONTROLLER',time_utc=now(),model_groups_paused=True))
            else:
                if mode=='STOP_EXPANSION_SAVE':(rd/'STOP_NEW_GPU').touch()
                write_json(rd/'guard-status.json',dict(state='CONTINUATION_ARMED',reason=mode,time_utc=now()))
        time.sleep(60)  # Local safety files only; no suite/trajectory scheduling.


if __name__=='__main__':main()
