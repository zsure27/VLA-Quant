"""Archive completed temporal micro and actual original videos, excluding active first50."""
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
    root = a.session / 'B1-temporal-v1'
    ev = a.control / 'B1-temporal-eval-v2'
    gate = root / 'eval-v2-micro-gate.json'
    g = json.loads(gate.read_text())
    if g['gate'] != 'PASS_B1_TEMPORAL_PAIRED_PROTOCOL' or g['paired_episodes']!=10:raise ValueError('Strict micro gate absent')
    files = {}
    def add(f,name):
        if f.is_symlink() or not f.is_file() or name in files:raise ValueError('Unsafe source')
        files[name]=f
    def tree(d,prefix):
        for f in sorted(d.rglob('*')):
            if f.is_file() and '__pycache__' not in f.parts:add(f,prefix+'/'+f.relative_to(d).as_posix())
    tree(root/'eval-v2-micro','micro')
    tree(root/'stage-records/audit-eval-v2-micro','audit')
    add(gate,'micro-gate.json');add(a.analysis,'analysis/'+a.analysis.name)
    for v in ('early80','phase80'):
        case=root/'eval-v2-micro'/v
        if (case/'exit-code.txt').read_text().strip()!='0':raise ValueError('Missing actual exit0')
        for n in ('artifact.json','adapter.pt'):
            add(root/v/'steps-1000'/n,'trained/'+v+'/'+n)
        log=next(case.glob('EVAL-*.txt'));import re
        stamp=re.search(r'\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}',log.name).group()
        vids=list((a.control.parent/'20261016-091-residual-vla-adaptation-v1/source-v1/rollouts').rglob(stamp+'--*.mp4'))
        if len(vids)!=10:raise ValueError('Original10 videos missing')
        for f in vids:add(f,'videos/'+v+'/'+f.name)
    if (root/'stage-records/audit-eval-v2-micro/exit-code.txt').read_text().strip()!='0':raise ValueError('Audit incomplete')
    for f in ev.glob('*.py'):add(f,'eval-source/'+f.name)
    for n in ('plan.json','runtime-lock.json','cpu-admission.json','launch.json'):
        add(ev/'first50-plan-v1'/n,'registered-plan/'+n)
    add(root/'train-1000-pair-gate.json','training-pair-gate.json')
    oldc=a.control.parent/'20261016-091-residual-vla-adaptation-v1'
    for n in ('source-v1.tar.gz','source-v1-SHA256.json','materials.json'):add(oldc/n,'frozen-parent/'+n)
    manifest = {'scope': 'Completed v2 temporal micro10 per candidate,20 original videos, raw queries, trained adapters and frozen scientific source. Complete training source/data/checkpoints in independently verified training archive; active first50 excluded. Full7B server checksum.',
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
    receipt={'gate':'PASS_SERVER_B1_TEMPORAL_MICRO_ARCHIVE','phase':'B1-temporal-v2-micro',
             'archive':str(a.archive),'bytes':archive.stat().st_size,'sha256':sha(archive),
             'files_verified':len(files),'members':len(seen),'videos':sum(n.endswith('.mp4') for n in files),
             'paired_episodes':10,'formal_episodes_added':0,'future_evaluations_excluded':True}
    (a.archive/'SERVER_VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
