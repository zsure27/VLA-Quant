"""Run only preregistered missing native references, with strict trace audit."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from qvla.infra.common import digest, write_json, now
from qvla.infra.lifecycle import quota_decision

p = argparse.ArgumentParser()
p.add_argument('--round-dir', type=Path, required=True)
p.add_argument('--suite', required=True)
a = p.parse_args()
series = json.loads((a.round_dir / 'series.json').read_text())
root = Path(series['persistent_root'])
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', XLA_PYTHON_CLIENT_PREALLOCATE='false',
           OMP_NUM_THREADS='4', JAX_PLATFORMS='cuda', MUJOCO_GL='egl')
for row in series['pending_cases']:
    if row['suite'] != a.suite:
        continue
    if quota_decision(json.loads((a.round_dir / 'quota.json').read_text()), series['control']) != 'CONTINUE':
        raise RuntimeError('No quota for another native episode')
    if shutil.disk_usage(root).free < 8 * 1024**3:
        raise RuntimeError('Continuation disk reserve reached')
    plan_path = a.round_dir / 'case-plans' / (row['round_id'] + '.json')
    plan = json.loads(plan_path.read_text())
    assert digest(plan_path) == row['plan_sha256']
    argv = [sys.executable, '-u', '-m', 'qvla.execute_pi05_bf16_episode_084_v2',
            '--checkpoint', str(root/'models/pi05_libero'),
            '--openpi-source', str(root/'sources/openpi'),
            '--libero-source', str(root/'sources/LIBERO'),
            '--libero-config', str(root/'libero-config-v1'),
            '--plan', str(plan_path), '--output', plan['output'], '--round-id', row['round_id'],
            '--suite', row['suite'], '--task-id', str(row['task_id']), '--reset-index', '5', '--model-seed', '7']
    with (a.round_dir/'case-logs'/(row['round_id']+'.log')).open('xb') as log:
        rc = subprocess.call(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
    write_json(a.round_dir/'case-status.json', {'time_utc':now(), 'case':row, 'exit_code':rc})
    if rc:
        raise RuntimeError('Native episode failed: '+row['round_id'])
    from qvla.audit_pi05_native_trace_v1 import audit
    report = audit(Path(plan['output']), plan_path)
    write_json(a.round_dir/'audits'/(row['round_id']+'.json'), report)
