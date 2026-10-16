"""Archive only a completed M0 phase, with exact original videos and no active shard."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('control', 'session', 'analysis', 'archive'):
        p.add_argument('--' + k, type=Path, required=True)
    p.add_argument('--phase', choices=('recovered-first50', 'reset-10-19', 'reset-20-29', 'reset-30-39', 'reset-40-49'), required=True)
    a = p.parse_args()
    c, r = a.control, a.session
    oldc = c.parent / '20261016-091-residual-vla-adaptation-v1'
    oldr = r.parent / '20261016-091-residual-vla-adaptation-v1'
    recovered = a.phase == 'recovered-first50'
    gatepath = r / ('first50-recovered-gate.json' if recovered else a.phase + '-gate.json')
    gate = json.loads(gatepath.read_text())
    required = 'PASS_B3_INIT_ONLY_SEGMENTED_PAIRED_PROTOCOL' if recovered else 'PASS_B3_INIT_ONLY_FULL_SHARD_PAIRED_PROTOCOL'
    if gate['gate'] != required or gate['paired_episodes'] != (50 if recovered else 100):
        raise ValueError('Full strict completed gate required')
    files = {}
    videos = []
    def add(f, name):
        if not f.is_file() or f.is_symlink() or name in files:
            raise ValueError('Missing/symlink/duplicate archive source: ' + str(f))
        files[name] = f
    def tree(root, prefix):
        for f in sorted(root.rglob('*')):
            if f.is_file() and '__pycache__' not in f.parts:
                add(f, prefix + '/' + f.relative_to(root).as_posix())
    if recovered:
        cases = [('ptq', oldr / 'first50/ptq', 50, 0),
                 ('svd-init-only-prefix47', oldr / 'first50/svd-init-only', 47, 0),
                 ('svd-init-only-missing3', r / 'missing3/svd-init-only', 3, 47)]
    else:
        cases = [(v, r / a.phase / v, 100, 0) for v in ('ptq', 'svd-init-only')]
    for tag, directory, n, offset in cases:
        interrupted = tag == 'svd-init-only-prefix47'
        if not interrupted and (directory / 'exit-code.txt').read_text().strip() != '0':
            raise ValueError('Real completed stage exit0 required')
        logs = list(directory.glob('EVAL-*.txt'))
        if len(logs) != 1:
            raise ValueError('Ambiguous log/video binding')
        stamp = re.search(r'\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}', logs[0].name).group()
        paths = sorted((oldc / 'source-v1/rollouts').rglob(stamp + '--*.mp4'))
        seen = set()
        variant = 'ptq' if tag == 'ptq' else 'svd-init-only'
        for f in paths:
            m = re.search(r'--episode=(\d+)--success=(True|False)--', f.name)
            if m is None or not 1 <= int(m[1]) <= n or int(m[1]) in seen or not f.stat().st_size:
                raise ValueError('Invalid video ordinal')
            if (m[2] == 'True') != gate['paired_rows'][offset + int(m[1]) - 1][variant]:
                raise ValueError('Video differs from paired outcome')
            seen.add(int(m[1]))
            name = 'videos/' + tag + '/' + f.name
            add(f, name); videos.append(name)
        if seen != set(range(1, n + 1)):
            raise ValueError('Video coverage missing')
        tree(directory, 'cases/' + tag)
    if len(videos) != gate['paired_episodes'] * 2:
        raise ValueError('Paired video total differs')
    add(gatepath, 'gate.json')
    add(a.analysis, 'ANALYSIS_CN.md')
    for n in ('source-v1.tar.gz', 'source-v1-SHA256.json', 'materials.json'):
        add(oldc / n, 'inherited-control/' + n)
    # Flat current controls are immutable helpers/proofs. Do not collect active
    # status/log trees from another phase.
    for f in sorted(c.iterdir()):
        if f.is_file() and (f.suffix in ('.py', '.json', '.md') or f.name == 'STOP_NEW_GPU'):
            add(f, 'control/' + f.name)
    if recovered:
        tree(c / 'recovery-plan-v2', 'control/recovery-plan-v2')
        for name in ('recovery-admission', 'audit-recovered-first50'):
            tree(r / 'stage-records' / name, 'wrapper/' + name)
        argv = json.loads((oldr / 'first50/svd-init-only/invocation.json').read_text())['command']
        for flag in ('--initial-state', '--initial-artifact'):
            f = Path(argv[argv.index(flag) + 1]); add(f, 'parent-materials/' + f.name)
    else:
        tree(c / 'M0-continuation-v2' / a.phase, 'control/' + a.phase)
        tree(r / 'stage-records' / ('audit-' + a.phase), 'wrapper/audit-' + a.phase)
    add(Path(__file__), 'archive-builder.py')
    manifest = {'phase': a.phase, 'complete_strict_gate': gate['gate'], 'original_videos': videos,
                'scope': 'Completed phase only; original interrupted suffix preserved as evidence, never counted; all active later phases excluded',
                'files': {name: {'source': str(f), 'bytes': f.stat().st_size, 'sha256': sha(f)} for name, f in sorted(files.items())}}
    raw = (json.dumps(manifest, indent=2) + '\n').encode()
    a.archive.mkdir(exist_ok=False)
    (a.archive / 'MANIFEST.json').write_bytes(raw)
    archive = a.archive / 'complete.tar.gz'
    with tarfile.open(archive, 'w:gz') as t:
        for name, f in sorted(files.items()):
            t.add(f, arcname=name)
        info = tarfile.TarInfo('MANIFEST.json'); info.size = len(raw)
        t.addfile(info, io.BytesIO(raw))
    count = 0
    with tarfile.open(archive, 'r|gz') as t:
        for m in t:
            h = hashlib.sha256()
            stream = t.extractfile(m)
            for b in iter(lambda: stream.read(4 * 1024**2), b''):
                h.update(b)
            expected = hashlib.sha256(raw).hexdigest() if m.name == 'MANIFEST.json' else manifest['files'][m.name]['sha256']
            if h.hexdigest() != expected:
                raise ValueError('Tar member changed: ' + m.name)
            count += 1
    if count != len(files) + 1:
        raise ValueError('Tar member coverage differs')
    receipt = {'gate': 'PASS_SERVER_M0_COMPLETED_PHASE_ARCHIVE', 'phase': a.phase,
               'archive': str(a.archive), 'bytes': archive.stat().st_size, 'sha256': sha(archive),
               'files_verified': len(files), 'videos': len(videos), 'members': count,
               'active_later_phases_excluded': True}
    (a.archive / 'SERVER_VERIFICATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
