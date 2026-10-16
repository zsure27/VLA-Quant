"""Archive preserved B1 temporal v1 protocol failure; zero admitted/formal queries."""
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
    b = a.control / 'B1-temporal-eval-v1'
    root = a.session / 'B1-temporal-v1/eval-micro/early80'
    status = json.loads((b / 'first50-plan-v1/control/20261017-091-B1-temporal80-micro-first50-v1/status.json').read_text())
    if status['state'] != 'FAILED' or status['last_exit_code'] != 1 or status['completed_stage_ids']:
        raise ValueError('Expected preserved v1 first-stage failure')
    events=[json.loads(x) for x in (root/'on-policy-events.jsonl').read_text().splitlines()]
    if any(x['record_type']=='query' for x in events):raise ValueError('Unexpected admitted queries')
    files = {}
    def add(f,name):
        if f.is_symlink() or not f.is_file() or name in files:raise ValueError('Unsafe source')
        files[name]=f
    def tree(d,prefix):
        for f in sorted(d.rglob('*')):
            if f.is_file() and '__pycache__' not in f.parts:add(f,prefix+'/'+f.relative_to(d).as_posix())
    tree(root,'failed-output');tree(b,'failed-control');add(a.analysis,'analysis/'+a.analysis.name)
    videos=list((a.control.parent/'20261016-091-residual-vla-adaptation-v1/source-v1/rollouts').rglob('2026_10_17-05_04_55--*.mp4'))
    if len(videos)!=10:raise ValueError('Expected10 aborted attempt videos')
    for f in videos:add(f,'aborted-videos/'+f.name)
    manifest = {'scope': 'Failed v1 micro attempt, all code/locked plan/output/10 aborted videos; zero formal episodes. Complete training archive is a separately indexed dependency',
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
    receipt={'gate':'PASS_SERVER_B1_TEMPORAL_FAILED_EVAL_ARCHIVE','phase':'B1-temporal-eval-v1-failed',
             'archive':str(a.archive),'bytes':archive.stat().st_size,'sha256':sha(archive),
             'files_verified':len(files),'members':len(seen),'videos':sum(n.endswith('.mp4') for n in files),
             'admitted_queries':0,'formal_episodes':0,'future_evaluations_excluded':True}
    (a.archive/'SERVER_VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
