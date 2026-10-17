"""Versioned complete resume-v2 set: immutable stage tars plus uncaptured files."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import tarfile


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--research-bundle', type=Path, required=True)
    p.add_argument('--final', action='store_true')
    a = p.parse_args()
    root = Path('/root/autodl-tmp/qvla-repro')
    c = root/'control/20261016-091-residual-resume-v2'
    oldc = c.parent/'20261016-091-residual-vla-adaptation-v1'
    r = root/'backups/experiments/p2-shared-peft'/c.name
    oldr = r.parent/oldc.name
    assert socket.gethostname() == 'autodl-container-dfbc4bb0ff-1e85dd94'
    assert a.archive.parent == root/'backups' and not a.archive.exists()
    if a.final:
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        for f in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                cmd = f.read_bytes()
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            assert not (b'vla_stage_supervisor.py' in cmd or b'run_eval_b1_temporal_091_v2.py' in cmd), 'Experiment still active'
    idx = json.loads((c/'COMPLETED_STAGE_BACKUP_INDEX_V2.json').read_text())
    a.archive.mkdir()
    covered = {}
    stages = []
    items = [(v['phase'], v) for v in idx['entries']]
    items += [('b1_temporal_training', idx['b1_temporal_training'])]
    items += [(k, v) for k, v in idx.items() if isinstance(v, dict) and v.get('gate') == 'PASS_LOCAL_RESIDUAL_SESSION_ARCHIVE']
    items += [(v['name'], dict(v, server=str(root/'backups'/v['name']), all_members_verified=True)) for v in idx['prior_reference_archives']]
    for key, item in items:
        if item.get('gate') == 'PASS_LOCAL_RESIDUAL_SESSION_ARCHIVE':
            assert item['members_verified'] == item['members']
        else:
            assert item['all_members_verified']
        d = Path(item.get('archive', item.get('server')))
        assert d.parent == root/'backups'
        src = d/'complete.tar.gz'
        assert sha(src) == item['sha256'] and src.stat().st_size == item['bytes']
        manifest = json.loads((d/'MANIFEST.json').read_text())
        for v in manifest['files'].values():
            covered.setdefault(v['source'], set()).add(v['sha256'])
        rel = 'stage-archives/'+d.name+'.tar.gz'
        target = a.archive/rel
        target.parent.mkdir(exist_ok=True)
        os.link(src, target)
        stages.append(dict(key=key, set_path=rel, **item))
    assert stages
    supplement = {}
    coverage = {}
    unstable = []
    def add(f, name):
        if f.is_symlink() or not f.is_file() or '__pycache__' in f.parts or f.suffix == '.pyc':
            return
        before = f.stat()
        raw = f.read_bytes()
        after = f.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            if a.final:
                raise ValueError('Final source changed: '+str(f))
            unstable.append(str(f))
            return
        h = hashlib.sha256(raw).hexdigest()
        if h in covered.get(str(f), set()):
            coverage[str(f)] = h
            return
        info = tarfile.TarInfo(name)
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
        supplement[name] = {'source': str(f), 'bytes': len(raw), 'sha256': h}
    with tarfile.open(a.archive/'complete.tar.gz', 'w:gz') as tar:
        for d, prefix in ((c, 'control'), (r, 'session'), (oldc, 'prior-control'), (oldr, 'prior-session')):
            for f in sorted(d.rglob('*')):
                rel = f.relative_to(d)
                if d == oldc and rel.parts[0] == 'source-v1' and (len(rel.parts)<2 or rel.parts[1] != 'rollouts'):
                    continue
                add(f, prefix+'/'+rel.as_posix())
        manifest = {'final': a.final, 'time_utc': datetime.now(timezone.utc).isoformat(), 'files': supplement,
                    'stage_coverage': coverage, 'unstable_files_excluded': unstable,
                    'scope': 'All current/prior-run raw session and control plus all original rollout videos; immutable complete stage dependencies included. Active files are a point-in-time snapshot, never a completed cohort. Full7B remains server only with existing checksum identity.'}
        raw = (json.dumps(manifest, indent=2)+'\n').encode()
        info = tarfile.TarInfo('MANIFEST.json'); info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
    (a.archive/'MANIFEST.json').write_bytes(raw)
    seen = set()
    with tarfile.open(a.archive/'complete.tar.gz', 'r|gz') as tar:
        for m in tar:
            assert m.isfile() and m.name not in seen
            data = tar.extractfile(m).read()
            expected = hashlib.sha256(raw).hexdigest() if m.name == 'MANIFEST.json' else supplement[m.name]['sha256']
            assert hashlib.sha256(data).hexdigest() == expected
            seen.add(m.name)
    assert seen == set(supplement)|{'MANIFEST.json'}
    for src, dst in ((c/'closure-materials-v2/VLA-Quant-852d350.bundle', 'VLA-Quant-852d350.bundle'), (a.research_bundle, 'research.bundle')):
        subprocess.run(['git', '-C', '/root/VLA-Quant', 'bundle', 'verify', str(src)], check=True, stdout=subprocess.DEVNULL)
        os.link(src, a.archive/dst)
    (a.archive/'SET_INDEX.json').write_text(json.dumps({'final': a.final, 'stage_archives': stages, 'manifest_sha256': sha(a.archive/'MANIFEST.json'), 'supplement_files': len(supplement), 'covered_source_files': len(coverage), 'unstable_files_excluded': unstable}, indent=2)+'\n')
    receipt = {'gate': 'PASS_SERVER_RESIDUAL_RESUME_SET', 'archive': str(a.archive), 'final': a.final,
               'bytes': (a.archive/'complete.tar.gz').stat().st_size, 'sha256': sha(a.archive/'complete.tar.gz'),
               'files_verified': len(supplement), 'members': len(seen), 'videos': sum(n.endswith('.mp4') for n in supplement),
               'complete_stage_archives': len(stages), 'unstable_files_excluded': unstable}
    (a.archive/'SERVER_VERIFICATION.json').write_text(json.dumps(receipt, indent=2)+'\n')
    files = sorted(f for f in a.archive.rglob('*') if f.is_file())
    sums = ''.join(sha(f)+'  '+f.relative_to(a.archive).as_posix()+'\n' for f in files)
    (a.archive/'SHA256SUMS.txt').write_text(sums)
    (a.archive/'RESULTS_SHA256SUMS.txt').write_text(sha(a.archive/'complete.tar.gz')+'  complete.tar.gz\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
