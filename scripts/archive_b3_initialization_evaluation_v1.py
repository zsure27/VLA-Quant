"""Archive completed Z/S micro and first50, including external original videos."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import tarfile
from archive_b3_initialization_training_v1 import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('control', 'session', 'archive', 'analysis'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--expected-hostname', required=True)
    p.add_argument('--phase', choices=('reset-10-19', 'reset-20-29', 'reset-30-39', 'reset-40-49'))
    p.add_argument('--hardlink-staging', action='store_true', help='Completed immutable files only; compressed tar remains the portable copy')
    p.add_argument('--completed-final-idle', action='store_true', help='Bounded last-phase archive only after all450 pairs and GPU termination')
    a = p.parse_args()
    phases = ((a.phase, 100),) if a.phase else (('micro', 10), ('first50', 50))
    if socket.gethostname() != a.expected_hostname or a.archive.exists():
        raise ValueError('Wrong host or existing archive; preserve prior attempt')
    if a.completed_final_idle:
        if a.phase != 'reset-40-49' or not a.hardlink_staging:
            raise ValueError('Idle final archive requires last phase and immutable hardlink staging')
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            raise ValueError('GPU must be empty for completed-final-idle admission')
        rows = []
        for phase_name in ('first50', 'reset-10-19', 'reset-20-29', 'reset-30-39', 'reset-40-49'):
            gate = json.loads((a.session / (phase_name + '-gate.json')).read_text())
            if gate['gate'] != 'PASS_B3_INITIALIZATION_PAIRED_PROTOCOL':
                raise ValueError('All completed phase gates required')
            rows.extend(gate['paired_rows'])
        if len(rows) != 450 or {(x['task_id'], x['init_state_index']) for x in rows} != {(t, k) for t in range(10) for k in range(5, 50)}:
            raise ValueError('Exact completed450 coverage required')
        # Already-compressed raw files dominate this stage. Account for the complete
        # last-stage tree, both exact video stamps, source/analysis, tar headers and
        # compression overhead before admitting the CPU archive with a 6GiB reserve.
        estimate_files = set(f for f in (a.session / a.phase).rglob('*') if f.is_file())
        estimate_files.update(f for f in a.analysis.rglob('*') if f.is_file())
        for variant in ('standard-zero', 'response-svd'):
            log = next((a.session / a.phase / variant).glob('EVAL-*.txt'))
            stamp = re.search(r'\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}', log.name).group()
            estimate_files.update((a.control / 'source-v2/rollouts').rglob(stamp + '--*.mp4'))
        source_manifest = json.loads((a.control / 'source-v2-SHA256.json').read_text())
        estimate_files.update(a.control / 'source-v2' / n for n in source_manifest)
        bound = int(sum(f.stat().st_size for f in estimate_files) * 1.02) + 64 * 1024**2
        if shutil.disk_usage(a.session).free - bound < 6 * 1024**3:
            raise ValueError('Final idle archive cannot preserve 6GiB after conservative size bound')
    elif shutil.disk_usage(a.session).free < 8 * 1024**3:
        raise ValueError('Preserve running reserve plus archive space')
    gates = {phase: json.loads((a.session / (phase + '-gate.json')).read_text())
             for phase, _ in phases}
    for phase, n in phases:
        if (gates[phase]['gate'] != 'PASS_B3_INITIALIZATION_PAIRED_PROTOCOL'
                or gates[phase]['paired_episodes'] != n):
            raise ValueError('Completed paired gate required')
    a.archive.mkdir(parents=True)
    origins = {}
    def add(source, relative):
        source = Path(source)
        target = a.archive / relative
        if not source.is_file() or target.exists():
            raise ValueError('Missing source or duplicate: ' + str(source))
        target.parent.mkdir(parents=True, exist_ok=True)
        if a.hardlink_staging:
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        else:
            shutil.copy2(source, target)
        origins[relative] = {'source': str(source), 'sha256': digest(target), 'bytes': target.stat().st_size}
    def tree(source, relative):
        source = Path(source)
        if not source.is_dir():
            raise ValueError('Missing completed directory: ' + str(source))
        for f in sorted(source.rglob('*')):
            if f.is_file() and '__pycache__' not in f.parts:
                add(f, relative + '/' + f.relative_to(source).as_posix())
    video_rows = []
    for phase, n in phases:
        add(a.session / (phase + '-gate.json'), 'payload/gates/' + phase + '-gate.json')
        for variant in ('standard-zero', 'response-svd'):
            case = a.session / phase / variant
            if (case / 'exit-code.txt').read_text().strip() != '0':
                raise ValueError('Completed case exit must be zero')
            logs = list(case.glob('EVAL-*.txt'))
            if len(logs) != 1:
                raise ValueError('Unique evaluation log required')
            stamp = re.search(r'\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}', logs[0].name).group()
            videos = list((a.control / 'source-v2/rollouts').rglob(stamp + '--*.mp4'))
            if len(videos) != n:
                raise ValueError('Original video count mismatch')
            ordinals = set()
            for video in videos:
                match = re.search(r'--episode=(\d+)--success=(True|False)--', video.name)
                if match is None or video.stat().st_size == 0:
                    raise ValueError('Invalid original video')
                ordinal = int(match[1])
                if ordinal in ordinals or not 1 <= ordinal <= n:
                    raise ValueError('Duplicate or invalid video ordinal')
                ordinals.add(ordinal)
                row = gates[phase]['paired_rows'][ordinal - 1]
                if (match[2] == 'True') != bool(row[variant]):
                    raise ValueError('Video success disagrees with audited episode')
                relative = 'payload/videos/' + phase + '/' + variant + '/' + video.name
                add(video, relative)
                video_rows.append({'phase': phase, 'variant': variant, 'ordinal': ordinal,
                                   **row, 'archive_path': relative, 'sha256': origins[relative]['sha256']})
            tree(case, 'payload/evaluation/' + phase + '/' + variant)
        wrapper = a.session / 'stage-records' / (phase + '-audit' if a.phase else 'audit-' + phase)
        if (wrapper / 'exit-code.txt').read_text().strip() != '0':
            raise ValueError('Audit wrapper exit must be zero')
        tree(wrapper, 'payload/wrapper-records/audit-' + phase)
    manifest = json.loads((a.control / 'source-v2-SHA256.json').read_text())
    if len(manifest) != 358:
        raise ValueError('Frozen source manifest changed')
    for name, expected in manifest.items():
        source = a.control / 'source-v2' / name
        if digest(source) != expected:
            raise ValueError('Frozen source hash mismatch: ' + name)
        add(source, 'payload/source-v2/' + name)
    for name in ('source-v2-SHA256.json', 'materials-b3-initialization-v1.json',
                 'training-update-diagnostic-v1.json', 'training-archive-scope-supplement-v2.json'):
        add(a.control / name, 'payload/control/' + name)
    for name in ('first50-plan.json', 'runtime-lock.json'):
        add(a.control / 'B3-plans-v1' / name, 'payload/control/B3-plans-v1/' + name)
    status = a.control / 'B3-plans-v1/control/20261015-046-quant-finetuning-v1-first50/status.json'
    state = json.loads(status.read_text())
    if len(state['completed_stage_ids']) != 12:
        raise ValueError('Original plan not fully complete')
    add(status, 'payload/control/completed-first50-status.json')
    if a.phase:
        directory = a.control / 'B3-continuation-v1' / a.phase
        for name in ('plan.json', 'runtime-lock.json', 'registration.json', 'launch.json'):
            add(directory / name, 'payload/control/' + a.phase + '/' + name)
        status = directory / 'control' / ('20261015-046-quant-finetuning-v1-' + a.phase) / 'status.json'
        state = json.loads(status.read_text())
        if len(state['completed_stage_ids']) != 3 or state['last_exit_code'] != 0:
            raise ValueError('Continuation not fully complete')
        add(status, 'payload/control/' + a.phase + '/completed-status.json')
    tree(a.analysis, 'payload/analysis')
    (a.archive / 'VIDEO_MANIFEST.json').write_text(json.dumps(video_rows, indent=2) + '\n')
    (a.archive / 'MANIFEST.json').write_text(json.dumps(origins, indent=2) + '\n')
    scope = {'time_utc': datetime.now(timezone.utc).isoformat(), 'hostname': socket.gethostname(),
             'scope': 'Complete Z/S raw traces, observations/actions, commands, exits, gates, original external videos, exact 358 frozen source files and analysis',
             'phases': [name for name, _ in phases], 'original_videos': len(video_rows),
             'staging_mode': 'hardlinks where possible; not an independent physical server replica' if a.hardlink_staging else 'copied files',
             'excluded_active': 'All evaluations outside explicitly listed completed phases',
             'prior_first50_archive': '/root/autodl-tmp/qvla-repro/backups/046-b3-initialization-first50-complete-20261015-v1',
             'prior_first50_tar_sha256': '9e3b38020e743c083c6712a72cdaf8fd48e6c9134b187fc1802c429c0ebd1952',
             'training_archive': '/root/autodl-tmp/qvla-repro/backups/046-b3-initialization-training-complete-20261015-v1',
             'training_tar_sha256': 'd0956ebefb3adb03d5b4b1cb057c5281e208b54a4b946b4cd109f1dd9121bf39',
             'prior_archive': '/root/autodl-tmp/qvla-repro/backups/091-gpu-b5-final-20261013-v1',
             'checkpoint_scope': 'Complete 7B checkpoint remains server-side; training/prior archives retain input and model provenance. No complete local 7B copy claim.'}
    (a.archive / 'SCOPE.json').write_text(json.dumps(scope, indent=2) + '\n')
    files = sorted(f for f in a.archive.rglob('*') if f.is_file())
    sums = {f.relative_to(a.archive).as_posix(): digest(f) for f in files}
    sums_path = a.archive / 'SHA256SUMS.txt'
    sums_path.write_text(''.join(f'{h}  {name}\n' for name, h in sums.items()))
    tar_path = a.archive / 'complete.tar.gz'
    with tarfile.open(tar_path, 'w:gz', compresslevel=1) as tar:
        for f in files + [sums_path]:
            tar.add(f, arcname=f.relative_to(a.archive).as_posix(), recursive=False)
    expected = {**sums, 'SHA256SUMS.txt': digest(sums_path)}
    seen = set()
    with tarfile.open(tar_path, 'r:gz') as tar:
        for member in tar:
            stream = tar.extractfile(member)
            if stream is None or member.name not in expected or member.name in seen:
                raise ValueError('Unexpected archive member')
            h = hashlib.sha256()
            for block in iter(lambda: stream.read(4 * 1024**2), b''):
                h.update(block)
            if h.hexdigest() != expected[member.name]:
                raise ValueError('Compressed member SHA mismatch')
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError('Compressed archive incomplete')
    receipt = {'gate': 'PASS_COMPLETED_B3_EVALUATION_SERVER_ARCHIVE', 'archive': str(a.archive),
               'archive_file': str(tar_path), 'bytes': tar_path.stat().st_size, 'sha256': digest(tar_path),
               'sha_entries': len(sums), 'payload_files': len(origins), 'original_videos': len(video_rows),
               'all_tar_members_verified': True, 'excluded_active_evaluation': True,
               'free_bytes_after': shutil.disk_usage(a.archive).free}
    (a.archive / 'SERVER_VERIFICATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
