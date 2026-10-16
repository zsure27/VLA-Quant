"""Download an immutable session tar and verify every member without duplicating it."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True)
    p.add_argument('--destination', type=Path, required=True)
    a = p.parse_args()
    if not a.name.startswith('091-residual-') or not a.name.replace('-', '').isalnum():
        raise ValueError('Unexpected archive name')
    repo = Path(__file__).resolve().parents[1]
    a.destination.mkdir(parents=True, exist_ok=False)
    options = ['-P', '18575', '-i', 'C:/Users/zsure/.ssh/id_ed25519_vla_014',
               '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o',
               'UserKnownHostsFile=' + (repo / '.local/control/20261016-091-residual-vla-adaptation-v1/known-hosts-091').as_posix()]
    remote = 'root@connect.nmb1.seetacloud.com:/root/autodl-tmp/qvla-repro/backups/' + a.name + '/'
    for n in ('SERVER_VERIFICATION.json', 'MANIFEST.json', 'complete.tar.gz'):
        subprocess.run(['scp', *options, remote + n, str(a.destination / n)], check=True)
    receipt = json.loads((a.destination / 'SERVER_VERIFICATION.json').read_text())
    manifest = json.loads((a.destination / 'MANIFEST.json').read_text())
    archive = a.destination / 'complete.tar.gz'
    h = hashlib.sha256()
    with archive.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    if h.hexdigest() != receipt['sha256'] or archive.stat().st_size != receipt['bytes']:
        raise ValueError('Downloaded archive differs from server')
    seen = set()
    with tarfile.open(archive, 'r|gz') as t:
        for m in t:
            parts = PurePosixPath(m.name)
            if not m.isfile() or parts.is_absolute() or '..' in parts.parts or m.name in seen:
                raise ValueError('Unsafe/duplicate archive member')
            seen.add(m.name)
            f = t.extractfile(m)
            h = hashlib.sha256()
            for b in iter(lambda: f.read(4 * 1024**2), b''):
                h.update(b)
            expected = (hashlib.sha256((a.destination / 'MANIFEST.json').read_bytes()).hexdigest()
                        if m.name == 'MANIFEST.json' else manifest['files'][m.name]['sha256'])
            if h.hexdigest() != expected:
                raise ValueError('Member SHA mismatch: ' + m.name)
    if seen != set(manifest['files']) | {'MANIFEST.json'}:
        raise ValueError('Incomplete member set')
    result = dict(receipt, gate='PASS_LOCAL_RESIDUAL_SESSION_ARCHIVE', members_verified=len(seen),
                  local_directory=str(a.destination), verification='All decompressed member SHA checked; tar retained, no duplicate extraction')
    (a.destination / 'LOCAL_VERIFICATION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
