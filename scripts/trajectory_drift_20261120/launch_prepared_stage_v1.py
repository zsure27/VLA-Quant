"""Explicit controller launch of one immutable prepared stage, never a scheduler."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    args = parser.parse_args()
    root, code, stage = args.control.resolve(), args.code.resolve(), args.stage.resolve()
    assert stage.parent == root / 'stages'
    sys.path[:0] = [str(code), str(root)]
    import pi05_duquant_catalog_20261120_v1  # Register before validating its plan.
    from qvla.infra.common import canonical, now, write_json
    from qvla.infra.plan import verify_plan
    from primary_only_closure_policy_v1 import decision
    auth = json.loads((root / 'authorization.json').read_text())
    quota = json.loads((root / 'quota.json').read_text())
    assert auth['hostname'] == socket.gethostname()
    assert datetime.datetime.fromisoformat(auth['expires_at']) > datetime.datetime.now(datetime.timezone.utc)
    assert decision(quota) == 'CONTINUE' and not (root / 'STOP_NEW_GPU').exists()
    assert json.loads((root / 'guard-status.json').read_text())['state'] == 'CONTINUATION_ARMED'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip() == auth['gpu_uuid']
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    assert shutil.disk_usage(root).free >= 8 * 1024**3
    assert all(not (stage / name).exists() for name in ('status.json', 'launch-receipt.json', 'execution.lock'))
    plan = json.loads((stage / 'plan.json').read_text())
    assert plan['plan_sha256'] == args.expected_plan_sha256
    verify_plan(plan)
    config = json.loads((stage / 'config.json').read_text())
    assert config['control']['server_id'] == auth['server_id']
    assert config['control']['shutdown_provider'] == 'user_pause'
    write_json(stage / 'authorization.json', dict(auth, round_id=stage.name, config_sha256=canonical(config), shutdown_authorized=False, pause_only=True))
    write_json(stage / 'quota.json', quota)
    argv = [sys.executable, '-u', '-m', 'qvla.infra', '--plugin', 'pi05_duquant_catalog_20261120_v1', 'run', str(stage / 'plan.json'), '--authorization', str(stage / 'authorization.json'), '--round-id', stage.name, '--quota', str(stage / 'quota.json'), '--execute']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', JAX_PLATFORMS='cpu', OMP_NUM_THREADS='4', PYTHONPATH=str(code)+':'+str(root), WANDB_MODE='disabled', TOKENIZERS_PARALLELISM='false')
    with (stage / 'supervisor.log').open('xb') as log:
        child = subprocess.Popen(argv, cwd=code, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(supervisor_pid=child.pid, argv=argv, started_at=now(), plan_sha256=plan['plan_sha256'])
    write_json(stage / 'launch-receipt.json', receipt)
    write_json(root / 'CURRENT_RUN.json', dict(stage=str(stage), control_code=str(code), condition=config['quantization']))
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
