"""Preserve the completed 091 experiment in a verifiable persistent archive."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path('/root/autodl-tmp/qvla-repro')
SOURCE = ROOT / 'backups/experiments/p2-shared-peft/20261009-091-b4-pair300-python-v4'
DEST = ROOT / 'backups/b4-091-final-20261010-0001'
CODE = ROOT / 'worktrees/20261009-091-b4-final100-resume'
COMMIT = 'f2309df3c4352fb3ddc8179955a5c875208e798e'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if not SOURCE.is_dir() or not CODE.is_dir():
        raise RuntimeError('Expected source/code missing')
    if subprocess.check_output(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], text=True).strip() != COMMIT:
        raise RuntimeError('Code commit mismatch')
    DEST.mkdir(exist_ok=True)
    data = DEST / 'experiment'
    shutil.copytree(SOURCE, data, copy_function=os.link, dirs_exist_ok=True)
    controls = DEST / 'controls'
    controls.mkdir(exist_ok=True)
    for name in ('20261009-091-b4-pair300-python-v4', '20261009-091-b4-final100-resume-v1'):
        src = ROOT / 'control' / name
        if not src.is_dir():
            raise RuntimeError(f'Missing control: {src}')
        shutil.copytree(src, controls / name, copy_function=os.link, dirs_exist_ok=True)
    bundle = DEST / f'VLA-Quant-{COMMIT}.bundle'
    if not bundle.exists():
        subprocess.run(['git', '-C', str(CODE), 'bundle', 'create', str(bundle), '--all'], check=True)
    scope = {'server_commit': COMMIT, 'source': str(SOURCE), 'plan_controls': [p.name for p in controls.iterdir()],
             'data_scope': 'same-task development reset20-49; no offline holdout', 'archive_method': 'hard links; source retained'}
    (DEST / 'SERVER_ARCHIVE_SCOPE.json').write_text(json.dumps(scope, indent=2) + '\n')
    rows = []
    for file in sorted(DEST.rglob('*')):
        if file.is_file() and file.name not in ('SHA256SUMS.txt', 'RESULTS_SHA256SUMS.txt'):
            rows.append(f'{digest(file)}  {file.relative_to(DEST).as_posix()}')
    (DEST / 'SHA256SUMS.txt').write_text('\n'.join(rows) + '\n')
    (DEST / 'RESULTS_SHA256SUMS.txt').write_text('\n'.join(r for r in rows if 'experiment/' in r) + '\n')
    print(json.dumps({'archive': str(DEST), 'files': len(rows), 'source': str(SOURCE), 'commit': COMMIT}))


if __name__ == '__main__':
    main()
