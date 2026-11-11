"""Publish this round's small evidence only; raw traces remain private."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from qvla.infra.common import write_json, digest, now

p = argparse.ArgumentParser()
p.add_argument('--round-dir', type=Path, required=True)
a = p.parse_args()
repo = Path(__file__).resolve().parents[2]
task = repo / 'Ex_Reports/_tasks/multisuite-trajectory-deviation-v1'
dest = task / 'rounds' / a.round_dir.name
dest.mkdir(parents=True, exist_ok=False)
for name in ['clone-verification.json', 'series.json', 'config.json', 'plan.json', 'status.json']:
    value = json.loads((a.round_dir / name).read_text())
    write_json(dest / name, value)
completed = []
for file in sorted((a.round_dir / 'audits').glob('*.json')):
    value = json.loads(file.read_text())
    completed.append(value)
write_json(dest / 'summary.json', {'time_utc': now(), 'new_native_episodes': len(completed),
    'native_successes': sum(x['success'] for x in completed), 'audits': completed,
    'quantized_trajectories': 0, 'quantization_bridge_admitted': False,
    'note': 'Native reference completion is not evidence of quantization drift.'})
files = [f for f in dest.glob('*') if f.is_file()]
marker = re.compile(rb'-----BEGIN .*PRIVATE KEY|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}')
for f in files:
    if f.stat().st_size > 2*1024**2 or marker.search(f.read_bytes()):
        raise ValueError('Public evidence security/size gate failed')
env = dict(os.environ, GIT_TERMINAL_PROMPT='0')
def git(*args):
    return subprocess.check_output(['git', '-C', str(repo), *args], env=env,
                                   text=True, stderr=subprocess.PIPE, timeout=60).strip()
try:
    git('add', '--', *[str(f.relative_to(repo)) for f in files])
    if git('diff', '--cached', '--name-only'):
        git('commit', '-m', 'Record 084 native reference continuation')
    head = git('rev-parse', 'HEAD')
    branch = 'refs/heads/codex/084-pi05-resume-20261111'
    url = 'https://github.com/zsure27/VLA-Quant.git'
    git('push', url, 'HEAD:' + branch)
    if git('ls-remote', url, branch).split()[0] != head:
        raise RuntimeError('Public remote SHA mismatch')
    write_json(a.round_dir / 'git-sync-receipt.json', {'commit': head, 'remote_verified': True})
except Exception as error:
    write_json(a.round_dir / 'cloud-sync-gap.json', {'time_utc':now(), 'error_type':type(error).__name__,
        'public_git_verified':False, 'persistent_originals_retained':True,
        'reason':'Server publication attempt failed; controller must complete sync.'})
    print('Public Git sync failed; persistent originals retained', flush=True)
    raise SystemExit(1)
