"""Verify the complete046 portable archive set and restore its final supplement."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def safe(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe archive-set path')
    return p.parts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--repository', type=Path, required=True)
    a = p.parse_args()
    display = a.directory.resolve()
    root = Path('\\\\?\\' + str(display)) if os.name == 'nt' else display
    server = json.loads((root / 'SERVER_SET_VERIFICATION.json').read_text())
    index = json.loads((root / 'ARCHIVE_SET_INDEX.json').read_text())
    if server['gate'] != 'PASS_SERVER_COMPLETE_ARCHIVE_SET_SHA_COVERAGE' or sha(root / 'ARCHIVE_SET_INDEX.json') != server['index_sha256']:
        raise ValueError('Trusted server final index mismatch')
    if len(index['stages']) != 6 or index['original_videos_verified'] != 920 or index['frozen_source_files_verified'] != 358:
        raise ValueError('Incomplete final set coverage')
    for row in index['stages']:
        original = Path(row['local'])
        local = json.loads((original / 'LOCAL_VERIFICATION.json').read_text())
        if not local['all_files_verified'] or local['sha256'] != row['sha256']:
            raise ValueError('Earlier complete local stage verification absent')
        destination = root.joinpath(*safe(row['set_path']))
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            try:
                os.link(original / 'complete.tar.gz', destination)
            except OSError:
                shutil.copy2(original / 'complete.tar.gz', destination)
        if destination.stat().st_size != row['bytes'] or sha(destination) != row['sha256']:
            raise ValueError('Local stage portable tar mismatch')
    sums = {name: h for h, name in (line.split('  ', 1) for line in (root / 'SHA256SUMS.txt').read_text().splitlines())}
    if len(sums) != server['set_files']:
        raise ValueError('Final file count mismatch')
    for name, h in sums.items():
        if sha(root.joinpath(*safe(name))) != h:
            raise ValueError('Final set SHA mismatch: ' + name)
    supplement = root / 'session-supplement.tar.gz'
    with tarfile.open(supplement, 'r:gz') as tar:
        checksum_bytes = tar.extractfile('SUPPLEMENT_SHA256SUMS.txt').read()
    expected = {name: h for h, name in (line.split('  ', 1) for line in checksum_bytes.decode().splitlines())}
    expected['SUPPLEMENT_SHA256SUMS.txt'] = hashlib.sha256(checksum_bytes).hexdigest()
    restored = root / 'supplement-restored'
    seen = set()
    with tarfile.open(supplement, 'r|gz') as tar:
        for member in tar:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError('Unexpected supplement member')
            destination = restored.joinpath(*safe(member.name))
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                with destination.open('xb') as stream:
                    shutil.copyfileobj(tar.extractfile(member), stream, length=4 * 1024**2)
            if sha(destination) != expected[member.name]:
                raise ValueError('Local final supplement SHA failed')
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError('Final supplement restore incomplete')
    repo = str(a.repository.resolve())
    bundles = (root / ('VLA-Quant-' + index['root_git_short_head'] + '.bundle'),
               root / ('VLA-Quant-research-' + index['research_commit'] + '.bundle'))
    for bundle in bundles:
        subprocess.run(['git', '-c', 'safe.directory=' + repo, '-c', 'core.longpaths=true', '-C', repo, 'bundle', 'verify', str(display / bundle.name)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    heads = subprocess.check_output(['git', 'bundle', 'list-heads', str(display / bundles[1].name)], text=True)
    if index['research_commit'] not in {line.split()[0] for line in heads.splitlines()}:
        raise ValueError('Final research Git commit missing')
    receipt = {'gate': 'PASS_LOCAL_COMPLETE_ARCHIVE_SET_SHA_COVERAGE',
               'time_utc': datetime.now(timezone.utc).isoformat(), 'directory': str(display),
               'server_directory': index['server_directory'], 'all_set_files_sha_verified': True,
               'all_session_files_covered': True, 'all_stage_archives_locally_restored_and_verified': True,
               'original_videos_verified': 920, 'frozen_source_files_verified': 358,
               'supplement_members_restored_and_verified': len(seen), 'stages': 6,
               'root_git_bundle_verified': True, 'research_git_bundle_verified': True,
               'research_commit': index['research_commit'], 'index_sha256': server['index_sha256'],
               'scope': index['scope'], 'checkpoint_scope': index['checkpoint_scope']}
    (root / 'LOCAL_SET_VERIFICATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
