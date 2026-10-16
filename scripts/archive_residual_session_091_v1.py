"""Snapshot this session without changing scientific outputs; final requires idle/pass."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import socket
import subprocess
import tarfile


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('control', 'session', 'archive'):
        p.add_argument('--' + n, type=Path, required=True)
    p.add_argument('--final', action='store_true')
    p.add_argument('--interrupted', action='store_true')
    a = p.parse_args()
    if socket.gethostname() != 'autodl-container-dfbc4bb0ff-1e85dd94':
        raise ValueError('Wrong instance')
    status = a.control / 'M0-plans-v1/control/20261016-091-residual-vla-adaptation-v1-M0-first50/status.json'
    observed = json.loads(status.read_text())
    if a.final:
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            raise ValueError('Final archive requires GPU idle')
        if a.interrupted:
            stop = json.loads((a.control / 'quota-stop-receipt.json').read_text())
            if stop['gpu_processes'] or stop['formal_pair_complete']:
                raise ValueError('Invalid interrupted closure receipt')
        else:
            gate = json.loads((a.session / 'first50-gate.json').read_text())
            if gate['gate'] != 'PASS_B3_INIT_ONLY_PAIRED_PROTOCOL' or gate['paired_episodes'] != 50:
                raise ValueError('Final archive requires completed first50 gate')
    a.archive.mkdir(exist_ok=False)
    files = {}
    def tree(root, prefix, source=False):
        for f in sorted(root.rglob('*')):
            if not f.is_file() or f.is_symlink() or '__pycache__' in f.parts:
                continue
            rel = f.relative_to(root)
            if source and rel.parts[0] == 'source-v1' and (len(rel.parts) < 2 or rel.parts[1] != 'rollouts'):
                continue  # Frozen sources are retained in source-v1.tar.gz plus SHA manifest.
            files[prefix + '/' + rel.as_posix()] = f
    tree(a.control, 'control', True)
    tree(a.session, 'session')
    parent = Path('/root/autodl-tmp/qvla-repro/backups/experiments/p2-shared-peft/20261015-046-quant-finetuning-v1/response-svd/steps-1000')
    for n in ('initial-adapter.pt', 'artifact.json', 'teacher-targets.pt', 'teacher-input-hashes.json'):
        f = parent / n
        if f.is_file():
            files['parent-materials/' + n] = f
    manifest = {'time_utc': datetime.now(timezone.utc).isoformat(), 'final': a.final, 'interrupted': a.interrupted,
                'scope': 'Current session raw outputs, videos, control, frozen source and parent initial factors; inherited full 7B remains server only.',
                'observed_status': observed, 'files': {}, 'unstable_files_excluded': []}
    archive = a.archive / 'complete.tar.gz'
    with tarfile.open(archive, 'w:gz') as t:
        for name, f in sorted(files.items()):
            before = f.stat()
            raw = f.read_bytes()
            after = f.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                if a.final:
                    raise ValueError('Final source is changing: ' + str(f))
                manifest['unstable_files_excluded'].append(str(f))
                continue
            manifest['files'][name] = {'source': str(f), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            t.addfile(info, io.BytesIO(raw))
        raw = (json.dumps(manifest, indent=2) + '\n').encode()
        info = tarfile.TarInfo('MANIFEST.json')
        info.size = len(raw)
        t.addfile(info, io.BytesIO(raw))
    (a.archive / 'MANIFEST.json').write_bytes(raw)
    with tarfile.open(archive, 'r|gz') as t:
        count = 0
        for m in t:
            if m.name == 'MANIFEST.json':
                continue
            h = hashlib.sha256()
            f = t.extractfile(m)
            for b in iter(lambda: f.read(4 * 1024**2), b''):
                h.update(b)
            if h.hexdigest() != manifest['files'][m.name]['sha256']:
                raise ValueError('Archived member mismatch')
            count += 1
    receipt = {'gate': 'PASS_SERVER_RESIDUAL_SESSION_ARCHIVE', 'final': a.final,
               'archive': str(a.archive), 'bytes': archive.stat().st_size, 'sha256': digest(archive),
               'files_verified': count, 'videos': sum(n.endswith('.mp4') for n in manifest['files']),
               'unstable_files_excluded': manifest['unstable_files_excluded'], 'observed_status': observed}
    (a.archive / 'SERVER_VERIFICATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
