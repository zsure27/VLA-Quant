"""Receive the supplement, reuse complete local stage tars, verify whole set SHA."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--destination', type=Path, required=True)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(repo/'scripts/receive_residual_archive_091_v1.py'), '--name', a.name, '--destination', str(a.destination)], check=True)
    options = ['-P', '18575', '-i', 'C:/Users/zsure/.ssh/id_ed25519_vla_014', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile='+str(repo/'.local/control/20261016-091-residual-vla-adaptation-v1/known-hosts-091')]
    remote = 'root@connect.nmb1.seetacloud.com:/root/autodl-tmp/qvla-repro/backups/'+a.name+'/'
    for n in ('SET_INDEX.json', 'SHA256SUMS.txt', 'RESULTS_SHA256SUMS.txt', 'VLA-Quant-852d350.bundle', 'research.bundle'):
        subprocess.run(['scp', *options, remote+n, str(a.destination/n)], check=True)
    index = json.loads((a.destination/'SET_INDEX.json').read_text())
    for item in index['stage_archives']:
        name = Path(item.get('archive', item.get('server'))).name
        candidates = [Path(item.get('local_directory', item.get('local', '')))/'complete.tar.gz', Path('D:/VLA-Backups/20261016-091-residual-resume-v2')/name/'complete.tar.gz', repo/'.local/archives/20261016-091-residual-vla-adaptation'/name/'complete.tar.gz']
        src = next((f for f in candidates if f.is_file() and sha(f)==item['sha256']), None)
        assert src, 'Missing verified local stage: '+name
        target = a.destination/item['set_path']; target.parent.mkdir(exist_ok=True)
        shutil.copyfile(src, target)
    count = 0
    for line in (a.destination/'SHA256SUMS.txt').read_text().splitlines():
        h, n = line.split(maxsplit=1)
        f = (a.destination/n).resolve()
        assert a.destination.resolve() in f.parents and sha(f)==h, n
        count += 1
    result = {'gate': 'PASS_LOCAL_RESIDUAL_RESUME_SET', 'final': index['final'], 'local_directory': str(a.destination), 'files_verified': count, 'stage_archives': len(index['stage_archives']), 'supplement_members_verified': json.loads((a.destination/'LOCAL_VERIFICATION.json').read_text())['members_verified'], 'manifest_sha256': index['manifest_sha256'], 'all_set_sha256_verified': True}
    (a.destination/'LOCAL_SET_VERIFICATION.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
