"""Portable archive of a completed PTQ/S0 phase, including original videos."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for z in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(z)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('control', 'session', 'archive', 'analysis'):
        p.add_argument('--' + n, type=Path, required=True)
    p.add_argument('--phase', choices=('micro', 'first50'), required=True)
    a = p.parse_args()
    g = json.loads((a.session / (a.phase + '-gate.json')).read_text())
    if g['gate'] != 'PASS_B3_INIT_ONLY_PAIRED_PROTOCOL':
        raise ValueError('Completed strict paired gate required')
    n = 10 if a.phase == 'micro' else 50
    if g['paired_episodes'] != n:
        raise ValueError('Wrong phase coverage')
    files = {}
    def add(path, name):
        if not path.is_file() or path.is_symlink() or name in files:
            raise ValueError('Missing/ambiguous archived file: ' + str(path))
        files[name] = path
    def tree(root, prefix):
        for f in sorted(root.rglob('*')):
            if f.is_file() and '__pycache__' not in f.parts:
                add(f, prefix + '/' + f.relative_to(root).as_posix())
    videos = []
    for variant in ('ptq', 'svd-init-only'):
        case = a.session / a.phase / variant
        if (case / 'exit-code.txt').read_text().strip() != '0':
            raise ValueError('Case is not complete')
        logs = list(case.glob('EVAL-*.txt'))
        if len(logs) != 1:
            raise ValueError('Expected one complete log')
        stamp = re.search(r'\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}', logs[0].name).group()
        paths = sorted((a.control / 'source-v1/rollouts').rglob(stamp + '--*.mp4'))
        if len(paths) != n:
            raise ValueError('Original video coverage mismatch')
        seen = set()
        for f in paths:
            m = re.search(r'--episode=(\d+)--success=(True|False)--', f.name)
            if m is None or not 1 <= int(m[1]) <= n or int(m[1]) in seen or f.stat().st_size == 0:
                raise ValueError('Invalid video ordinal')
            seen.add(int(m[1]))
            if (m[2] == 'True') != bool(g['paired_rows'][int(m[1]) - 1][variant]):
                raise ValueError('Video outcome differs from strict gate')
            name = 'videos/' + variant + '/' + f.name
            add(f, name)
            videos.append({'variant': variant, 'ordinal': int(m[1]), 'path': name})
        tree(case, 'evaluation/' + variant)
    wrapper = a.session / 'stage-records' / ('audit-' + a.phase)
    if (wrapper / 'exit-code.txt').read_text().strip() != '0':
        raise ValueError('Audit wrapper not complete')
    tree(wrapper, 'wrapper/audit-' + a.phase)
    add(a.session / (a.phase + '-gate.json'), 'gate.json')
    add(a.analysis, 'ANALYSIS_CN.md')
    for name in ('source-v1.tar.gz', 'source-v1-SHA256.json', 'materials.json',
                 'M0-plans-v1/first50-plan.json', 'M0-plans-v1/runtime-lock.json'):
        add(a.control / name, 'control/' + name)
    add(Path(__file__), 'archive-builder.py')
    a.archive.mkdir(exist_ok=False)
    manifest = {'scope': 'Complete ' + a.phase + ' only; other active phases excluded',
                'videos': videos,
                'files': {n: {'source': str(f), 'bytes': f.stat().st_size, 'sha256': sha(f)}
                          for n, f in sorted(files.items())}}
    raw = (json.dumps(manifest, indent=2) + '\n').encode()
    (a.archive / 'MANIFEST.json').write_bytes(raw)
    archive = a.archive / 'complete.tar.gz'
    with tarfile.open(archive, 'w:gz') as t:
        for name, f in sorted(files.items()):
            t.add(f, arcname=name)
        info = tarfile.TarInfo('MANIFEST.json')
        info.size = len(raw)
        t.addfile(info, io.BytesIO(raw))
    with tarfile.open(archive) as t:
        if len(t.getmembers()) != len(files) + 1:
            raise ValueError('Tar member coverage mismatch')
        for name, z in manifest['files'].items():
            if hashlib.sha256(t.extractfile(name).read()).hexdigest() != z['sha256']:
                raise ValueError('Tar member SHA mismatch')
    receipt = {'gate': 'PASS_SERVER_COMPLETED_INIT_ONLY_PHASE_ARCHIVE',
               'phase': a.phase, 'archive': str(a.archive), 'bytes': archive.stat().st_size,
               'sha256': sha(archive), 'files_verified': len(files), 'members': len(files) + 1,
               'original_videos': len(videos), 'other_active_phases_excluded': True}
    (a.archive / 'SERVER_VERIFICATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
