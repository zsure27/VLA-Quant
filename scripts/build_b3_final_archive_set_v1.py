"""Build the complete current046 archive set without recompressing verified stages."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tarfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def tar_metadata(path, name):
    with tarfile.open(path, 'r|gz') as tar:
        for item in tar:
            if item.name == name:
                return json.loads(tar.extractfile(item).read())
    raise ValueError('Missing tar metadata: ' + name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('control', 'session', 'stage-index', 'documents', 'research-bundle', 'archive'):
        p.add_argument('--' + n, type=Path, required=True)
    p.add_argument('--research-commit', required=True)
    a = p.parse_args()
    if socket.gethostname() != 'autodl-container-90824aab47-6dc00ae3' or a.archive.exists():
        raise ValueError('Wrong046 or existing final archive')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise ValueError('GPU must be empty')
    final = json.loads((a.control / 'analysis-final450-v1/FINAL450_ANALYSIS.json').read_text())
    if final['gate'] != 'PASS_FINAL450_PAIRED_REVIEW' or final['formal_episodes_each'] != 450:
        raise ValueError('Final450 review required')
    index = json.loads(a.stage_index.read_text())
    if len(index['archives']) != 6:
        raise ValueError('Exactly six complete stage archives required')
    a.archive.mkdir(parents=True)
    covered = {}
    stages = []
    video_sources = {}
    for item in index['archives']:
        directory = Path(item['server'])
        tar_path = directory / 'complete.tar.gz'
        receipt = json.loads((directory / 'SERVER_VERIFICATION.json').read_text())
        if (sha(tar_path) != item['sha256'] or tar_path.stat().st_size != item['bytes']
                or receipt['sha256'] != item['sha256'] or not receipt['all_tar_members_verified']
                or not item['all_tar_members_and_local_files_verified']):
            raise ValueError('Stage archive anchor mismatch')
        manifest = tar_metadata(tar_path, 'MANIFEST.json')
        for entry in manifest.values():
            covered.setdefault(entry['source'], set()).add(entry['sha256'])
        if 'training-complete' not in directory.name:
            for video in tar_metadata(tar_path, 'VIDEO_MANIFEST.json'):
                origin = manifest[video['archive_path']]['source']
                if origin in video_sources and video_sources[origin] != video['sha256']:
                    raise ValueError('Conflicting original video SHA')
                video_sources[origin] = video['sha256']
        relative = 'stage-archives/' + directory.name + '.tar.gz'
        target = a.archive / relative
        target.parent.mkdir(exist_ok=True)
        os.link(tar_path, target)
        stages.append({**item, 'set_path': relative})
    if len(video_sources) != 920 or any(sha(Path(s)) != h for s, h in video_sources.items()):
        raise ValueError('Exact920 original video SHA coverage failed')
    frozen = json.loads((a.control / 'source-v2-SHA256.json').read_text())
    if len(frozen) != 358 or any(sha(a.control / 'source-v2' / n) != h for n, h in frozen.items()):
        raise ValueError('Frozen358 source SHA failed')
    supplement = {}
    coverage = {}
    roots = ((a.session, 'session'), (a.control, 'control'), (a.documents, 'documents'))
    for root, prefix in roots:
        for f in sorted(root.rglob('*')):
            if not f.is_file() or '__pycache__' in f.parts or f.suffix == '.pyc':
                continue
            # Final documents and both bundles are handled explicitly, once.
            if prefix == 'control' and (a.documents in f.parents or f.suffix == '.bundle'
                                        or f.name.startswith('build-final-archive-set')):
                continue
            h = sha(f)
            archive_path = prefix + '/' + f.relative_to(root).as_posix()
            is_covered = h in covered.get(str(f), set())
            coverage[str(f)] = {'sha256': h, 'covered_by_stage': is_covered,
                                'supplement_path': None if is_covered else archive_path}
            if not is_covered:
                supplement[archive_path] = {'source': str(f), 'sha256': h, 'bytes': f.stat().st_size}
    manifest_bytes = (json.dumps(supplement, indent=2) + '\n').encode()
    supplement_hashes = {name: row['sha256'] for name, row in supplement.items()}
    supplement_hashes['SUPPLEMENT_MANIFEST.json'] = hashlib.sha256(manifest_bytes).hexdigest()
    checksum_bytes = ''.join(f'{h}  {name}\n' for name, h in supplement_hashes.items()).encode()
    target = a.archive / 'session-supplement.tar.gz'
    with tarfile.open(target, 'w:gz', compresslevel=1, dereference=True) as tar:
        for name, row in supplement.items():
            tar.add(row['source'], arcname=name, recursive=False)
        for name, data in (('SUPPLEMENT_MANIFEST.json', manifest_bytes), ('SUPPLEMENT_SHA256SUMS.txt', checksum_bytes)):
            info = tarfile.TarInfo(name); info.size = len(data); tar.addfile(info, io.BytesIO(data))
    expected = {**supplement_hashes, 'SUPPLEMENT_SHA256SUMS.txt': hashlib.sha256(checksum_bytes).hexdigest()}
    seen = set()
    with tarfile.open(target, 'r|gz') as tar:
        for member in tar:
            if not member.isfile() or member.name in seen or member.name not in expected:
                raise ValueError('Unsafe/unexpected supplement member')
            h = hashlib.sha256()
            stream = tar.extractfile(member)
            for b in iter(lambda: stream.read(4 * 1024**2), b''):
                h.update(b)
            if h.hexdigest() != expected[member.name]:
                raise ValueError('Supplement member SHA failed')
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError('Supplement incomplete')
    root_git = subprocess.check_output(['git', '-C', '/root/VLA-Quant', 'rev-parse', '--short', 'HEAD'], text=True).strip()
    root_bundle = a.control / 'closure-materials-v1' / ('VLA-Quant-' + root_git + '.bundle')
    for original, name in ((root_bundle, root_bundle.name), (a.research_bundle, 'VLA-Quant-research-' + a.research_commit + '.bundle')):
        subprocess.run(['git', '-C', '/root/VLA-Quant', 'bundle', 'verify', str(original)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        os.link(original, a.archive / name)
    heads = subprocess.check_output(['git', 'bundle', 'list-heads', str(a.research_bundle)], text=True)
    if a.research_commit not in {line.split()[0] for line in heads.splitlines()}:
        raise ValueError('Current research Git commit absent from bundle')
    result = {'time_utc': datetime.now(timezone.utc).isoformat(), 'server_directory': str(a.archive),
              'stages': stages, 'original_videos_verified': len(video_sources), 'frozen_source_files_verified': len(frozen),
              'current_file_coverage': coverage, 'supplement_files': len(supplement),
              'research_commit': a.research_commit, 'root_git_short_head': root_git,
              'scope': 'Complete current046 initialization run training+micro+900formal episodes; all920 original videos; stage tars plus supplement; hardlinks not new physical replicas',
              'checkpoint_scope': 'Full7B checkpoint remains server-side with existing full checksums; no complete local7B copy claim',
              'prior_091_archive': index['prior_091_archive'], 'prior_091_sha256': index['prior_091_sha256'],
              'post_closure_receipts': 'Native shutdown and Windows requests are append-only receipts after final data seal'}
    (a.archive / 'ARCHIVE_SET_INDEX.json').write_text(json.dumps(result, indent=2) + '\n')
    all_files = sorted(f for f in a.archive.rglob('*') if f.is_file())
    sums = {f.relative_to(a.archive).as_posix(): sha(f) for f in all_files}
    (a.archive / 'SHA256SUMS.txt').write_text(''.join(f'{h}  {n}\n' for n, h in sums.items()))
    (a.archive / 'RESULTS_SHA256SUMS.txt').write_text(''.join(f'{h}  {n}\n' for n, h in sums.items() if n.endswith('.tar.gz')))
    receipt = {'gate': 'PASS_SERVER_COMPLETE_ARCHIVE_SET_SHA_COVERAGE', 'server_directory': str(a.archive),
               'set_files': len(sums), 'set_bytes': sum(f.stat().st_size for f in all_files),
               'all_set_files_sha_verified': True, 'all_session_files_covered': True,
               'original_videos_verified': 920, 'source_files_verified': 358,
               'supplement_tar_members_verified': len(seen), 'research_commit': a.research_commit,
               'index_sha256': sha(a.archive / 'ARCHIVE_SET_INDEX.json'), 'free_bytes_after': shutil.disk_usage(a.archive).free}
    (a.archive / 'SERVER_SET_VERIFICATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
