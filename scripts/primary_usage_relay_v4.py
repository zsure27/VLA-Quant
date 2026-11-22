"""Local quota-only safety relay. Never reads episode logs or selects experiments.

Official account RPC every five minutes, with bounded failure and exact run/boot
binding. Credentials remain in this process environment; no auth export to GPU.
"""
import argparse,datetime,hashlib,json,os,sys,time
from pathlib import Path
from read_primary_usage_app_server_v1 import read_primary

def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(temporary,path)

def deliver(binding,quota):
    import paramiko
    client=paramiko.SSHClient();client.load_host_keys(binding['known_hosts']);client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(binding['host'],port=binding['port'],username='root',password=os.environ[binding['password_env']],look_for_keys=False,allow_agent=False,timeout=20,banner_timeout=20,auth_timeout=20)
    try:
        expected=binding['host_key_sha256']
        import base64
        actual=base64.b64encode(hashlib.sha256(client.get_transport().get_remote_server_key().asbytes()).digest()).decode().rstrip('=')
        if actual!=expected:raise ValueError('Server host key differs')
        program='''from pathlib import Path
import json,os,hashlib,socket,datetime
b=json.loads(BINDING_LITERAL);q=json.loads(QUOTA_LITERAL);root=Path(b['control'])
assert socket.gethostname()==b['hostname']
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==b['boot_id']
assert hashlib.sha256((root/'authorization.json').read_bytes()).hexdigest()==b['authorization_sha256']
auth=json.loads((root/'authorization.json').read_text());assert auth['gpu_uuid']==b['gpu_uuid']
assert datetime.datetime.fromisoformat(auth['expires_at'])>datetime.datetime.now(datetime.timezone.utc)
assert str(root).encode() in Path('/proc',str(b['guard_pid']),'cmdline').read_bytes()
state=json.loads((root/'guard-status.json').read_text())['state']
assert state in ['CONTINUATION_ARMED','WAIT_API_REFRESH'], 'CONTROL_CLOSING_NO_WRITE'
observed=datetime.datetime.fromisoformat(q['observed_at'].replace('Z','+00:00'))
assert 0<=(datetime.datetime.now(datetime.timezone.utc)-observed).total_seconds()<90
assert q['window_minutes']==300 and q['source']=='codex_api' and 'weekly' not in q
for p in [root/'quota.json',*(root/'stages').glob('*/quota.json')]:
 assert json.loads((root/'guard-status.json').read_text())['state'] in ['CONTINUATION_ARMED','WAIT_API_REFRESH']
 t=p.with_suffix('.relay.tmp');t.write_text(json.dumps(q));os.replace(t,p)
print(json.dumps(dict(gate='PASS_FRESH_PRIMARY_DELIVERED_BOUND_RUN',observed_at=q['observed_at'],remaining_percent=q['remaining_percent'],boot_id=b['boot_id'],control=str(root))))
'''.replace('BINDING_LITERAL',repr(json.dumps(binding))).replace('QUOTA_LITERAL',repr(json.dumps(quota)))
        stdin,stdout,stderr=client.exec_command(binding['remote_python']+' -',timeout=30);stdin.write(program);stdin.channel.shutdown_write()
        result=stdout.read().decode('utf-8');error=stderr.read().decode('utf-8');rc=stdout.channel.recv_exit_status()
        if rc:raise RuntimeError('CONTROL_CLOSING_NO_WRITE' if 'CONTROL_CLOSING_NO_WRITE' in error else 'Remote bound-run delivery rejected')
        return json.loads(result)
    finally:client.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--state-dir',type=Path,required=True);p.add_argument('--interval',type=int,default=300);p.add_argument('--once',action='store_true');a=p.parse_args()
    assert 60<=a.interval<=300
    binding=json.loads(a.binding.read_text(encoding='utf-8'));a.state_dir.mkdir(parents=True,exist_ok=True)
    # The run must explicitly allow this quota-only process; no experiment scheduling.
    assert binding['quota_only'] is True and binding['server_id']=='084'
    write(a.state_dir/'process.json',dict(pid=os.getpid(),started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),binding_sha256=hashlib.sha256(a.binding.read_bytes()).hexdigest(),interval_seconds=a.interval,experiment_scheduler=False))
    failed_cycles=0;unavailable_since=None
    while not (a.state_dir/'STOP').exists():
        started=time.monotonic();ok=False
        for attempt in range(2):
            try:
                quota=read_primary(timeout=45,executable=binding['codex_executable']);receipt=deliver(binding,quota)
                record=dict(state='DELIVERED',quota=quota,receipt=receipt,attempt=attempt+1)
                write(a.state_dir/'latest.json',record)
                with (a.state_dir/'deliveries.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps(record)+'\n')
                ok=True;break
            except Exception as e:
                write(a.state_dir/'latest.json',dict(state='FAILED_NO_CACHED_SUBSTITUTION',error_type=type(e).__name__,reason=str(e)[:120],observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
                if 'CONTROL_CLOSING_NO_WRITE' in str(e) or 'Remote bound-run delivery rejected' in str(e) or 'Server host key differs' in str(e):
                    write(a.state_dir/'stopped.json',dict(reason='CURRENT_BINDING_CLOSED_OR_INVALID_NO_FURTHER_WRITES'))
                    return
                if attempt==0:time.sleep(20)
        if a.once:return 0 if ok else 1
        failed_cycles=0 if ok else failed_cycles+1
        if ok:unavailable_since=None
        elif unavailable_since is None:unavailable_since=started
        if unavailable_since is not None and time.monotonic()-unavailable_since>=900:
            write(a.state_dir/'stopped.json',dict(reason='FIFTEEN_MINUTES_UNAVAILABLE_REMOTE_STALE_GUARD_REMAINS_ENABLED',failed_cycles=failed_cycles))
            return 1
        # API failure never fabricates freshness or disables the remote stop guard.
        delay=max(1,(a.interval if ok else 60)-(time.monotonic()-started))
        if ok:
            # Poll just AFTER the API-reported window boundary, then require a real
            # fresh response. The clock never substitutes for API reset proof.
            delay=min(delay,max(1,quota['resets_at']-time.time()+2))
        until=time.monotonic()+delay
        while time.monotonic()<until and not (a.state_dir/'STOP').exists():time.sleep(max(0,min(2,until-time.monotonic())))
    write(a.state_dir/'stopped.json',dict(stopped_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),reason='CURRENT_RUN_STOP_MARKER'))

if __name__=='__main__':raise SystemExit(main())
