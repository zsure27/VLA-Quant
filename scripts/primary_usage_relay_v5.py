"""Quota-only relay with explicit lease preflight and typed remote outcomes.

Independent version: v4 and all existing run bindings remain untouched.
No experiment scheduling, lease renewal, cached quota, or threshold override.
"""
import argparse
import base64
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

from read_primary_usage_app_server_v1 import read_primary


class RelayRejected(RuntimeError):
    def __init__(self, code, retryable=False):
        super().__init__(code)
        self.code, self.retryable = code, retryable


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


def timestamp(value):
    result = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Timezone required')
    return result


def preflight(binding, required_runtime_seconds, closure_seconds, now=None):
    """Fail before GPU launch if the explicit lease cannot cover the run budget."""
    now = now or utc_now()
    if required_runtime_seconds <= 0 or closure_seconds < 600:
        raise ValueError('Positive runtime and at least 600 seconds closure required')
    if binding.get('quota_only') is not True:
        raise ValueError('Explicit quota-only binding required')
    expires = timestamp(binding['authorization_expires_at'])
    if not binding.get('authorization_deadline_source'):
        raise ValueError('Explicit operational deadline source required; no implicit 8h default')
    seconds = (expires - now).total_seconds()
    if seconds < required_runtime_seconds + closure_seconds:
        raise RelayRejected('LEASE_INSUFFICIENT_FOR_PLANNED_RUN_AND_CLOSURE')
    return dict(gate='PASS_QUOTA_RELAY_LOCAL_LEASE_PREFLIGHT',
                authorization_expires_at=expires.isoformat(),
                lease_remaining_seconds=seconds,
                required_runtime_seconds=required_runtime_seconds,
                closure_seconds=closure_seconds,
                remote_delivery_verified=False)


def validate_snapshot(binding, snapshot, quota, now):
    """Pure remote contract; never extends or rewrites authorization."""
    for field, code in [('hostname', 'HOSTNAME_CHANGED'), ('boot_id', 'BOOT_CHANGED'),
                        ('authorization_sha256', 'AUTHORIZATION_SHA_CHANGED'),
                        ('gpu_uuid', 'GPU_CHANGED')]:
        if snapshot[field] != binding[field]:
            raise RelayRejected(code)
    if not snapshot['guard_matches']:
        raise RelayRejected('GUARD_PROCESS_CHANGED_OR_EXITED')
    if snapshot['guard_state'] not in ('CONTINUATION_ARMED', 'WAIT_API_REFRESH'):
        raise RelayRejected('CONTROL_CLOSING_NO_WRITE')
    expires = timestamp(snapshot['authorization_expires_at'])
    if expires != timestamp(binding['authorization_expires_at']):
        raise RelayRejected('AUTHORIZATION_EXPIRY_CHANGED')
    if expires <= now:
        raise RelayRejected('AUTHORIZATION_EXPIRED')
    age = (now - timestamp(quota['observed_at'])).total_seconds()
    if not 0 <= age < 90:
        raise RelayRejected('QUOTA_STALE_OR_FUTURE', True)
    if (quota.get('source') != 'codex_api' or quota.get('window_minutes') != 300
            or set(quota) - {'source', 'transport', 'window_minutes', 'observed_at',
                             'remaining_percent', 'resets_at', 'rpc_latency_seconds'}
            or isinstance(quota.get('remaining_percent'), bool)
            or not isinstance(quota.get('remaining_percent'), (int, float))
            or not 0 <= quota['remaining_percent'] <= 100
            or not isinstance(quota.get('resets_at'), int)
            or quota['resets_at'] <= now.timestamp()):
        raise RelayRejected('INVALID_FRESH_PRIMARY', True)
    return (expires - now).total_seconds()


def remote_program(binding, quota):
    # Send only explicit contracts and data, never passwords or local auth files.
    shared = '\n\n'.join(inspect.getsource(x) for x in
                           (RelayRejected, timestamp, validate_snapshot))
    payload = repr(json.dumps(dict(binding=binding, quota=quota)))
    return '''import datetime,json,hashlib,os,socket,subprocess
from pathlib import Path
''' + shared + '\nPAYLOAD = ' + payload + '''
data=json.loads(PAYLOAD);b=data['binding'];q=data['quota'];root=Path(b['control'])
try:
    auth_bytes=(root/'authorization.json').read_bytes();auth=json.loads(auth_bytes)
    guard_path=Path('/proc',str(b['guard_pid']),'cmdline')
    snapshot=dict(hostname=socket.gethostname(),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        authorization_sha256=hashlib.sha256(auth_bytes).hexdigest(),
        gpu_uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).strip(),
        authorization_expires_at=auth['expires_at'],
        guard_matches=guard_path.exists() and str(root).encode() in guard_path.read_bytes(),
        guard_state=json.loads((root/'guard-status.json').read_text())['state'])
    remaining=validate_snapshot(b,snapshot,q,datetime.datetime.now(datetime.timezone.utc))
    for path in [root/'quota.json',*(root/'stages').glob('*/quota.json')]:
        if json.loads((root/'guard-status.json').read_text())['state'] not in ('CONTINUATION_ARMED','WAIT_API_REFRESH'):
            raise RelayRejected('CONTROL_CLOSING_NO_WRITE')
        temp=path.with_suffix('.relay.tmp');temp.write_text(json.dumps(q));os.replace(temp,path)
    print(json.dumps(dict(ok=True,code='PASS_FRESH_PRIMARY_DELIVERED_BOUND_RUN',
        observed_at=q['observed_at'],lease_remaining_seconds=remaining,
        lease_warning=remaining<=3600)))
except RelayRejected as error:
    print(json.dumps(dict(ok=False,code=error.code,retryable=error.retryable)))
except Exception:
    print(json.dumps(dict(ok=False,code='REMOTE_SNAPSHOT_OR_IO_UNAVAILABLE',retryable=True)))
'''


def decode_reply(reply):
    try:
        result = json.loads(reply)
        if not isinstance(result, dict) or type(result.get('ok')) is not bool:
            raise ValueError()
        if result['ok']:
            if result.get('code') != 'PASS_FRESH_PRIMARY_DELIVERED_BOUND_RUN':
                raise ValueError()
            return result
        code = result['code']
        allowed = {'HOSTNAME_CHANGED', 'BOOT_CHANGED', 'AUTHORIZATION_SHA_CHANGED',
                   'GPU_CHANGED', 'GUARD_PROCESS_CHANGED_OR_EXITED',
                   'CONTROL_CLOSING_NO_WRITE', 'AUTHORIZATION_EXPIRY_CHANGED',
                   'AUTHORIZATION_EXPIRED', 'QUOTA_STALE_OR_FUTURE',
                   'INVALID_FRESH_PRIMARY', 'REMOTE_SNAPSHOT_OR_IO_UNAVAILABLE'}
        if code not in allowed:
            raise ValueError()
        # Retry policy is controlled locally, never trusted to a remote boolean.
        raise RelayRejected(code, code in {'QUOTA_STALE_OR_FUTURE',
                            'INVALID_FRESH_PRIMARY', 'REMOTE_SNAPSHOT_OR_IO_UNAVAILABLE'})
    except (ValueError, KeyError, TypeError):
        raise RelayRejected('REMOTE_REPLY_UNAVAILABLE', True) from None


def deliver(binding, quota):
    import paramiko
    client = paramiko.SSHClient()
    client.load_host_keys(binding['known_hosts'])
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        client.connect(binding['host'], port=binding['port'], username='root',
                       password=os.environ[binding['password_env']], look_for_keys=False,
                       allow_agent=False, timeout=20, banner_timeout=20, auth_timeout=20)
        actual = base64.b64encode(hashlib.sha256(
            client.get_transport().get_remote_server_key().asbytes()).digest()).decode().rstrip('=')
        if actual != binding['host_key_sha256']:
            raise RelayRejected('HOST_KEY_CHANGED')
        stdin, stdout, stderr = client.exec_command(binding['remote_python']+' -', timeout=30)
        stdin.write(remote_program(binding, quota)); stdin.channel.shutdown_write()
        result = stdout.read().decode('utf-8')
        stderr.read()  # Never persist unstructured stderr or account/server data.
        if stdout.channel.recv_exit_status():
            raise RelayRejected('REMOTE_COMMAND_FAILED', True)
        return decode_reply(result)
    except (paramiko.BadHostKeyException, paramiko.AuthenticationException):
        raise RelayRejected('SSH_IDENTITY_OR_AUTHENTICATION_REJECTED') from None
    finally:
        client.close()


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binding', type=Path, required=True)
    p.add_argument('--state-dir', type=Path, required=True)
    p.add_argument('--required-runtime-seconds', type=int, required=True)
    p.add_argument('--closure-seconds', type=int, default=900)
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--once', action='store_true')
    args = p.parse_args()
    binding = json.loads(args.binding.read_text(encoding='utf-8'))
    args.state_dir.mkdir(parents=True, exist_ok=True)
    checked = preflight(binding, args.required_runtime_seconds, args.closure_seconds)
    write(args.state_dir/'preflight.json', checked)
    if args.preflight_only:
        print(json.dumps(checked)); return 0
    write(args.state_dir/'process.json', dict(pid=os.getpid(), started_at=utc_now().isoformat(),
        binding_sha256=hashlib.sha256(args.binding.read_bytes()).hexdigest(),
        interval_seconds=300, experiment_scheduler=False))
    unavailable_since = None
    while not (args.state_dir/'STOP').exists():
        started = time.monotonic(); success = False
        for attempt in range(2):
            try:
                quota = read_primary(timeout=45, executable=binding['codex_executable'])
                receipt = deliver(binding, quota)
                record = dict(state='DELIVERED', quota=quota, receipt=receipt)
                write(args.state_dir/'latest.json', record)
                with (args.state_dir/'deliveries.jsonl').open('a', encoding='utf-8') as log:
                    log.write(json.dumps(record)+'\n')
                if receipt['lease_warning']:
                    write(args.state_dir/'attention-required.json', dict(
                        code='LEASE_EXPIRES_WITHIN_ONE_HOUR_CONTROLLER_ACTION_REQUIRED',
                        observed_at=utc_now().isoformat(),
                        authorization_expires_at=binding['authorization_expires_at'],
                        automatic_renewal=False))
                success = True; break
            except Exception as error:
                code = error.code if isinstance(error, RelayRejected) else 'PRIMARY_OR_TRANSPORT_UNAVAILABLE'
                write(args.state_dir/'latest.json', dict(state='FAILED_NO_CACHED_SUBSTITUTION',
                    code=code, observed_at=utc_now().isoformat()))
                if isinstance(error, RelayRejected) and not error.retryable:
                    write(args.state_dir/'stopped.json', dict(code=code, observed_at=utc_now().isoformat()))
                    return 1
                if attempt == 0:
                    time.sleep(20)
        if args.once:
            return 0 if success else 1
        if success:
            unavailable_since = None
        elif unavailable_since is None:
            unavailable_since = started
        if unavailable_since is not None and time.monotonic()-unavailable_since >= 900:
            write(args.state_dir/'stopped.json', dict(code='FIFTEEN_MINUTES_UNAVAILABLE',
                                                     observed_at=utc_now().isoformat()))
            return 1
        delay = max(1, (300 if success else 60)-(time.monotonic()-started))
        if success:
            delay = min(delay, max(1, quota['resets_at']-time.time()+2))
        until = time.monotonic()+delay
        while time.monotonic() < until and not (args.state_dir/'STOP').exists():
            time.sleep(max(0, min(2, until-time.monotonic())))
    write(args.state_dir/'stopped.json', dict(code='CURRENT_RUN_STOP_MARKER', observed_at=utc_now().isoformat()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
