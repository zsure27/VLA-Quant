"""Reclaim only two copied staging payloads backed by pinned server/local tars."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import socket
import tarfile

ANCHORS = {
    '046-b3-initialization-first50-complete-20261015-v1': (399964959, '9e3b38020e743c083c6712a72cdaf8fd48e6c9134b187fc1802c429c0ebd1952'),
    '046-b3-reset-10-19-complete-20261015-v1': (679500413, '2cec7fcaf7059a61b18c8d25a6df97947801a61524fdc929b76a264a97e0da56'),
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--local-receipts', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if socket.gethostname() != 'autodl-container-90824aab47-6dc00ae3':
        raise ValueError('Wrong current instance')
    root = Path('/root/autodl-tmp/qvla-repro/backups').resolve(strict=True)
    reviewed = []
    for name, (size, expected_sha) in ANCHORS.items():
        archive = root / name
        payload = archive / 'payload'
        if archive.is_symlink() or payload.is_symlink() or payload.resolve(strict=True) != root / name / 'payload':
            raise ValueError('Resolved target outside exact named staging directory')
        tar_path = archive / 'complete.tar.gz'
        local = json.loads((a.local_receipts / (name + '.json')).read_text())
        server = json.loads((archive / 'SERVER_VERIFICATION.json').read_text())
        if (tar_path.stat().st_size != size or sha(tar_path) != expected_sha
                or local['sha256'] != expected_sha or not local['all_files_verified']
                or local['bytes'] != size or server['sha256'] != expected_sha
                or not server['all_tar_members_verified']):
            raise ValueError('Pinned independent archive or local receipt mismatch')
        with tarfile.open(tar_path, 'r:gz') as tar:
            manifest_bytes = tar.extractfile('MANIFEST.json').read()
        if manifest_bytes != (archive / 'MANIFEST.json').read_bytes():
            raise ValueError('Staging manifest differs from pinned tar')
        manifest = json.loads(manifest_bytes)
        expected_files = {}
        for relative, item in manifest.items():
            if not relative.startswith('payload/') or '..' in Path(relative).parts:
                raise ValueError('Unexpected manifest path')
            target = archive / relative
            original = Path(item['source'])
            if target.is_symlink() or original.is_symlink() or not original.is_file():
                raise ValueError('Original file missing or ambiguous')
            if sha(target) != item['sha256'] or sha(original) != item['sha256']:
                raise ValueError('Staging/original content mismatch')
            if target.stat().st_ino == original.stat().st_ino and target.stat().st_dev == original.stat().st_dev:
                raise ValueError('Only physically copied staging payloads are eligible')
            expected_files[target] = item['bytes']
        actual_files = {f for f in payload.rglob('*') if f.is_file()}
        if actual_files != set(expected_files) or any(f.is_symlink() for f in payload.rglob('*')):
            raise ValueError('Unrecorded or linked files inside deletion target')
        reviewed.append({'archive': str(archive), 'payload': str(payload), 'files': len(expected_files),
                         'payload_bytes': sum(expected_files.values()), 'tar_sha256': expected_sha,
                         'local_receipt_sha256': sha(a.local_receipts / (name + '.json'))})
    receipt = {'time_utc': datetime.now(timezone.utc).isoformat(), 'reviewed': reviewed,
               'execute': a.execute, 'free_bytes_before': shutil.disk_usage(root).free,
               'retained': 'Original source/evaluation data, pinned complete.tar.gz, manifest/checksums/receipts and verified local full copies'}
    if a.execute:
        for item in reviewed:
            payload = Path(item['payload'])
            if payload.resolve(strict=True) != root / Path(item['archive']).name / 'payload':
                raise ValueError('Final target recheck failed')
            shutil.rmtree(payload)
        receipt['free_bytes_after'] = shutil.disk_usage(root).free
        for item in reviewed:
            (Path(item['archive']) / 'STAGING_PAYLOAD_RECLAIMED_V1.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
