"""Audit one historical SVD spool; reclaim only after independent local backup.

No model is loaded. SHA256 is streamed. The default operation writes an audit
receipt and never deletes. A local backup verifier must emit the receipt schema
checked by validate_backup_receipt before --execute can be admitted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import socket
import stat
import subprocess


TARGET = Path('/root/autodl-tmp/qvla-repro/backups/experiments/p2-shared-peft/'
              '20261008-046-b2-b3-proprio-v4/B3/smoke10/svd-inputs')
MATERIALS = Path('/root/autodl-tmp/qvla-repro/control/'
                 '20261015-046-quant-finetuning-v1/materials-b3-initialization-v1.json')
PRESERVED = ('initial-adapter.pt', 'teacher-targets.pt', 'teacher-input-hashes.json', 'artifact.json')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(part)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True).encode()).hexdigest()


def safe_relative(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name or str(p) != name:
        raise ValueError('Unsafe or noncanonical relative path')
    return name


def require_exact_directory(path, expected):
    path, expected = Path(path), Path(expected)
    if path != expected or path.is_symlink() or path.resolve(strict=True) != expected:
        raise ValueError('Target must be the exact fixed historical spool directory')
    if not path.is_dir():
        raise ValueError('Spool directory is absent')
    for parent in path.parents:
        if parent.is_symlink():
            raise ValueError('Symlink ancestor is forbidden')


def stat_identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_nlink)


def file_record(path, relative=None):
    path = Path(path)
    before = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise ValueError('Only regular non-symlink files are accepted: ' + str(path))
    digest = sha(path)
    after = path.lstat()
    if stat_identity(before) != stat_identity(after):
        raise ValueError('File changed while hashing: ' + str(path))
    return {'relative_path': safe_relative(relative) if relative is not None else None,
            'path': str(path), 'bytes': after.st_size, 'sha256': digest,
            'device': after.st_dev, 'inode': after.st_ino, 'nlink': after.st_nlink,
            'mtime_ns': after.st_mtime_ns,
            'allocated_bytes': getattr(after, 'st_blocks', 0) * 512}


def inventory(target):
    records = []
    for directory, subdirs, filenames in os.walk(target, followlinks=False):
        root = Path(directory)
        if root.is_symlink() or any((root / name).is_symlink() for name in subdirs):
            raise ValueError('Symlink in spool directory tree')
        for name in sorted(filenames):
            p = root / name
            records.append(file_record(p, p.relative_to(target).as_posix()))
    records.sort(key=lambda r: r['relative_path'])
    if not records:
        raise ValueError('Empty spool is not a reclaim candidate')
    groups = {}
    for row in records:
        rel = PurePosixPath(row['relative_path'])
        if len(rel.parts) != 2 or rel.suffix != '.pt':
            raise ValueError('Unexpected file outside target/sample spool layout')
        groups.setdefault(rel.parts[0], set()).add(rel.name)
    expected_samples = {f'{i:03d}.pt' for i in range(80)}
    if len(groups) != 224 or any(names != expected_samples for names in groups.values()):
        raise ValueError('Expected exactly224 target directories with80 .pt samples each')
    return records


def content_manifest(target, records):
    return {'schema': 'vla-svd-spool-content-v1', 'target': str(target),
            'files': [{k: row[k] for k in ('relative_path', 'bytes', 'sha256')} for row in records]}


def validate_backup_receipt(receipt, manifest, hostname):
    if (receipt.get('gate') != 'PASS_LOCAL_SVD_SPOOL_BACKUP_ALL_FILES'
            or receipt.get('all_files_verified') is not True
            or receipt.get('independent_local_copy') is not True
            or receipt.get('server_hostname') != hostname
            or receipt.get('target') != manifest['target']
            or receipt.get('content_manifest_sha256') != canonical_sha(manifest)
            or receipt.get('files') != manifest['files']):
        raise ValueError('Independent local per-file backup receipt does not match current spool')
    local = receipt.get('local_archive_path', '')
    if (not local or local.startswith('/root/') or not receipt.get('archive_sha256')
            or len(receipt['archive_sha256']) != 64 or receipt.get('archive_bytes', 0) <= 0):
        raise ValueError('Receipt must identify a verified independent local archive')


def dependencies(materials_path):
    m = json.loads(materials_path.read_text())
    if Path(m['shared_b3_smoke_dir']) != TARGET.parent:
        raise ValueError('Materials do not bind the fixed historical smoke directory')
    retained = {name: file_record(TARGET.parent / name) for name in PRESERVED}
    artifact = json.loads((TARGET.parent / 'artifact.json').read_text())
    if artifact.get('model_id') != 'B3' or artifact.get('steps') != 10 or artifact.get('gate') != 'PASS_SMOKE10':
        raise ValueError('Historical initialization artifact gate mismatch')
    for name in PRESERVED[:-1]:
        if retained[name]['sha256'] != artifact.get('reuse_files', {}).get(name):
            raise ValueError('Preserved factor/teacher file differs from historical artifact: ' + name)
    inputs = {}
    for key, pattern in (('student80', 'obs-*.npz'), ('calibration80', 'sample-*.npz')):
        root = Path(m[key])
        # Student observations use the manifest's recorded filenames; accept its
        # actual flat NPZ names rather than imposing a historical naming prefix.
        paths = sorted(root.glob('*.npz') if key == 'student80' else root.glob(pattern))
        if len(paths) != 80 or root.is_symlink():
            raise ValueError(key + ' must retain exactly80 input files')
        inputs[key] = {'manifest': file_record(root / 'manifest.json'),
                       'files': [file_record(p, p.name) for p in paths]}
    if inputs['student80']['manifest']['sha256'] != artifact.get('training_manifest_sha256'):
        raise ValueError('Student80 manifest differs from the historical initialization artifact')
    profiles = {key: file_record(m[key]) for key in ('w2_g128', 'w2_g64', 'w4')}
    checkpoint_root = Path(m['checkpoint'])
    identity = artifact.get('checkpoint_identity', {})
    hashes = identity.get('files', {})
    if not hashes or not any(n.endswith(('.safetensors', '.bin')) for n in hashes):
        raise ValueError('Historical complete checkpoint identity is missing')
    checkpoint = {}
    for name, expected in sorted(hashes.items()):
        safe_relative(name)
        p = checkpoint_root / name
        if p.resolve(strict=True) != checkpoint_root.resolve(strict=True) / name:
            raise ValueError('Checkpoint dependency escaped its root')
        item = file_record(p, name)
        if item['sha256'] != expected:
            raise ValueError('Checkpoint dependency no longer matches historical SHA: ' + name)
        checkpoint[name] = item
    return {'materials': file_record(materials_path), 'parent_preserved': retained,
            'inputs': inputs, 'profiles': profiles,
            'trajectory_split': file_record(m['trajectory_split']),
            'checkpoint_root': str(checkpoint_root), 'checkpoint_identity': identity,
            'checkpoint_files_verified': checkpoint,
            'note': 'These dependencies remain untouched; backup admission does not depend on regenerability'}


def require_idle(target):
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                   '--format=csv,noheader'], text=True).strip()
    if gpu:
        raise ValueError('GPU processes are active; no reclamation is allowed')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            for fd in (proc / 'fd').iterdir():
                try:
                    value = os.readlink(fd).removesuffix(' (deleted)')
                except FileNotFoundError:
                    continue
                if value == str(target) or value.startswith(str(target) + '/'):
                    raise ValueError('A process holds the spool open: PID ' + proc.name)
        except (FileNotFoundError, ProcessLookupError):
            continue


def save_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-hostname', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--materials', type=Path, default=MATERIALS)
    p.add_argument('--local-backup-receipt', type=Path)
    p.add_argument('--expected-local-backup-receipt-sha256')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    hostname = socket.gethostname()
    if hostname != a.expected_hostname:
        raise ValueError('Wrong current instance hostname')
    require_exact_directory(TARGET, TARGET)
    output = a.output.resolve()
    if output == TARGET or TARGET in output.parents or output.exists():
        raise ValueError('Receipt must be new and outside the deletion target')
    if a.execute and (not a.local_backup_receipt or not a.expected_local_backup_receipt_sha256):
        raise ValueError('--execute requires a local receipt and its separately pinned SHA256')
    records = inventory(TARGET)
    manifest = content_manifest(TARGET, records)
    deps = dependencies(a.materials)
    unique = {(row['device'], row['inode']): row for row in records}
    links_inside = {}
    for row in records:
        key = (row['device'], row['inode'])
        links_inside[key] = links_inside.get(key, 0) + 1
    result = {'gate': 'PASS_SVD_SPOOL_AUDIT', 'time_utc': datetime.now(timezone.utc).isoformat(),
              'hostname': hostname, 'target': str(TARGET), 'execute': a.execute,
              'content_manifest_sha256': canonical_sha(manifest), 'content_manifest': manifest,
              'files': records, 'dependencies': deps, 'file_count': len(records),
              'logical_bytes': sum(r['bytes'] for r in records),
              'unique_allocated_bytes': sum(r['allocated_bytes'] for r in unique.values()),
              'estimated_reclaimable_allocated_bytes': sum(r['allocated_bytes'] for k, r in unique.items()
                                                         if r['nlink'] == links_inside[k]),
              'free_bytes_before': shutil.disk_usage(TARGET).free}
    if a.local_backup_receipt:
        receipt_path = a.local_backup_receipt.resolve(strict=True)
        if TARGET in receipt_path.parents or receipt_path.is_symlink():
            raise ValueError('Backup receipt cannot reside in the target')
        receipt_sha = sha(receipt_path)
        if a.expected_local_backup_receipt_sha256 and receipt_sha != a.expected_local_backup_receipt_sha256:
            raise ValueError('Pinned local receipt SHA mismatch')
        validate_backup_receipt(json.loads(receipt_path.read_text()), manifest, hostname)
        result['local_backup_receipt_sha256'] = receipt_sha
        result['independent_local_backup_verified'] = True
    if a.execute:
        require_idle(TARGET)
        require_exact_directory(TARGET, TARGET)
        current = inventory(TARGET)
        if current != records:
            raise ValueError('Spool changed after admission; no deletion performed')
        output.parent.mkdir(parents=True, exist_ok=True)
        save_new(Path(str(output) + '.admission.json'), result)
        with Path(str(output) + '.journal.jsonl').open('x', encoding='utf-8') as journal:
            for row in records:
                path = TARGET / row['relative_path']
                # Recheck exact location and file identity immediately before unlink.
                if path.resolve(strict=True) != path or file_record(path, row['relative_path']) != row:
                    raise ValueError('Deletion target changed; stop and retain journal')
                path.unlink()
                journal.write(json.dumps({'deleted': row['relative_path'], 'sha256': row['sha256']}) + '\n')
                journal.flush()
            os.fsync(journal.fileno())
        for directory in sorted((d for d in TARGET.rglob('*') if d.is_dir()),
                                key=lambda d: len(d.parts), reverse=True):
            if directory.is_symlink() or TARGET not in directory.parents:
                raise ValueError('Unsafe empty directory')
            directory.rmdir()
        TARGET.rmdir()
        if any(file_record(TARGET.parent / name) != deps['parent_preserved'][name] for name in PRESERVED):
            raise ValueError('Preserved parent material changed unexpectedly')
        result.update({'gate': 'PASS_SVD_SPOOL_RECLAIM_BACKED_BY_LOCAL_COPY',
                       'free_bytes_after': shutil.disk_usage(TARGET.parent).free,
                       'deleted_files': len(records), 'parent_materials_unchanged': True})
    output.parent.mkdir(parents=True, exist_ok=True)
    save_new(output, result)
    print(json.dumps({k: result[k] for k in ('gate', 'target', 'execute', 'file_count',
                                           'logical_bytes', 'content_manifest_sha256')}))


if __name__ == '__main__':
    main()
