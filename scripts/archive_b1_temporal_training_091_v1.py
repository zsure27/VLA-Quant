"""Archive only completed B1 temporal training/data and its original A4 input source."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import socket
import tarfile


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('control', 'session', 'archive', 'analysis'):
        p.add_argument('--' + n, type=Path, required=True)
    a = p.parse_args()
    if socket.gethostname() != 'autodl-container-dfbc4bb0ff-1e85dd94':
        raise ValueError('Wrong host')
    b = a.control / 'B1-temporal-v1'
    root = a.session / 'B1-temporal-v1'
    g = json.loads((root / 'train-1000-pair-gate.json').read_text())
    if g['gate'] != 'PASS_B1_TEMPORAL_TRAIN_PAIR':
        raise ValueError('Training incomplete')
    status = json.loads((b / 'training-plan-v1/control/20261017-091-B1-temporal80-training-v1/status.json').read_text())
    if status['last_exit_code'] != 0 or len(status['completed_stage_ids']) != 6:
        raise ValueError('Six actual stages must finish')
    files = {}
    def add(f, name):
        if f.is_symlink() or not f.is_file() or name in files:
            raise ValueError('Missing/symlink/duplicate source')
        files[name] = f
    def tree(directory, prefix):
        for f in sorted(directory.rglob('*')):
            if f.is_file() and '__pycache__' not in f.parts:
                add(f, prefix + '/' + f.relative_to(directory).as_posix())
    # Explicit completed roots: future eval directories are deliberately excluded.
    for name in ('data', 'early80', 'phase80', 'stage-records'):
        tree(root / name, 'training/' + name)
    for n in (10,1000):
        add(root / ('train-%d-pair-gate.json' % n), 'training/train-%d-pair-gate.json' % n)
    tree(b, 'control/B1-temporal-v1')
    add(a.analysis, 'analysis/' + a.analysis.name)
    oldc = a.control.parent / '20261016-091-residual-vla-adaptation-v1'
    for n in ('source-v1.tar.gz', 'source-v1-SHA256.json', 'materials.json'):
        add(oldc / n, 'frozen-parent/' + n)
    source = Path(json.loads((root / 'data/early80/manifest.json').read_text())['source_run'])
    tree(source, 'original-a4-training-capture')
    parent = Path(json.loads((b / 'early80-materials.json').read_text())['shared_b3_smoke_dir'])
    for n in ('artifact.json', 'initial-adapter.pt', 'teacher-targets.pt', 'teacher-input-hashes.json'):
        add(parent / n, 'parent-svd/' + n)
    manifest = {'scope': 'Completed temporal training+160 selected inputs+all1020 original A4 source queries+frozen code and parent factors; future evaluation excluded; full7B remains server checksum',
                'files': {n: {'source': str(f), 'bytes': f.stat().st_size, 'sha256': sha(f)} for n,f in sorted(files.items())}}
    a.archive.mkdir(exist_ok=False)
    raw = (json.dumps(manifest, indent=2)+'\n').encode()
    (a.archive / 'MANIFEST.json').write_bytes(raw)
    archive = a.archive / 'complete.tar.gz'
    with tarfile.open(archive, 'w:gz') as t:
        for name,f in sorted(files.items()):
            if sha(f) != manifest['files'][name]['sha256']:
                raise ValueError('Completed source changed')
            t.add(f, arcname=name, recursive=False)
        info=tarfile.TarInfo('MANIFEST.json');info.size=len(raw);t.addfile(info, io.BytesIO(raw))
    seen=set()
    with tarfile.open(archive, 'r|gz') as t:
        for m in t:
            h=hashlib.sha256();f=t.extractfile(m)
            for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
            expected=hashlib.sha256(raw).hexdigest() if m.name=='MANIFEST.json' else manifest['files'][m.name]['sha256']
            if not m.isfile() or m.name in seen or h.hexdigest()!=expected:
                raise ValueError('Tar member mismatch')
            seen.add(m.name)
    if seen != set(files)|{'MANIFEST.json'}:
        raise ValueError('Archive incomplete')
    receipt={'gate':'PASS_SERVER_B1_TEMPORAL_TRAINING_ARCHIVE','phase':'B1-temporal-training',
             'archive':str(a.archive),'bytes':archive.stat().st_size,'sha256':sha(archive),
             'files_verified':len(files),'members':len(seen),'videos':sum(n.endswith('.mp4') for n in files),
             'source_queries':1020,'selected_inputs':160,'future_evaluations_excluded':True}
    (a.archive/'SERVER_VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
