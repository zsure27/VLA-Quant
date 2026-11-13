"""Continuation of admitted collectors with exact references and fresh authorization.

CPU manifests are deliberately not executable plans. A future GPU round must
compile an infra plan and bind a new authorization, GPU UUID and API observation.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import numpy as np
from qvla.infra.common import canonical, digest, read_json, write_json, now
from qvla.infra.lifecycle import quota_decision


CONDITIONS = ('awq_w4a16', 'gptq_w4a16', 'omniquant_w4a16',
              'smoothquant_w4a8', 'smoothquant_w4a4', 'qvla_w4a16')


def validate_manifest(series):
    if series.get('format') != 'OFT_EXISTING_METHODS_INITIAL25_CPU_PREP_V1':
        raise ValueError('Wrong manifest format')
    if series.get('training') != 'none' or series.get('calibration_resets') != [0]:
        raise ValueError('Frozen no-training/calibration role differs')
    keys = [(x['condition'], x['suite'], x['task_id'], x['reset_index']) for x in series['cases']]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate pending case')
    for case in series['cases']:
        if case['condition'] not in CONDITIONS or case['reset_index'] != 25:
            raise ValueError('Unprepared condition or reset')
        if not case.get('reference_result') or len(case.get('reference_sha256', '')) != 64:
            raise ValueError('Missing same-case reference identity')


def require_current_authorization(auth, series, round_dir, config, hostname, gpu_uuid, current=None):
    current = current or datetime.now(timezone.utc)
    if series.get('executable') is not True or auth.get('gpu_execution_authorized') is not True:
        raise ValueError('CPU preparation is not GPU execution authorization')
    if auth.get('round_id') != Path(round_dir).name or auth.get('config_sha256') != canonical(config):
        raise ValueError('Authorization belongs to another round/config')
    if auth.get('hostname') != hostname or auth.get('gpu_uuid') != gpu_uuid:
        raise ValueError('Current host/GPU identity differs')
    granted = datetime.fromisoformat(auth['granted_at'].replace('Z', '+00:00'))
    prepared = datetime.fromisoformat(series['prepared_at'].replace('Z', '+00:00'))
    if not prepared <= granted <= current:
        raise ValueError('Authorization predates this preparation or is in the future')


def build_command(root, series, case, output, case_id):
    root = Path(root)
    condition = case['condition']
    spec = series['collectors'][condition]
    command = [str(root / 'envs/oft/bin/python'), '-u', spec['path']]
    if condition == 'qvla_w4a16':
        command += ['--root', str(root), '--gates', series['profiles'][condition]['path'],
                    '--admission', spec['admission'], '--reference', str(Path(case['reference_result']).parent),
                    '--suite', case['suite'], '--task-id', str(case['task_id']),
                    '--reset', '25', '--round-id', case_id, '--output', str(output)]
    else:
        method = condition.split('_w')[0]
        command += ['--oft-source', str(root / 'sources/openvla-oft'),
                    '--checkpoint', str(root / 'models/openvla_oft'),
                    '--suite', case['suite'], '--task-id', str(case['task_id']),
                    '--initial-state-index', '25', '--model-seed', '7',
                    '--round-id', case_id, '--output', str(output),
                    '--profile', series['profiles'][condition]['path'],
                    '--official-root', str(root / 'sources/official-quantization-pinned-v2'),
                    '--method', method, '--activation-bits', condition.rsplit('a', 1)[1]]
        if method in ('gptq', 'omniquant'):
            command += ['--calibration-dir', str(root / 'calibration/20261109-four-suite-reset0-v1')]
    return command


def first_input(folder):
    files = list(Path(folder).glob('episodes/*/query-00000.npz'))
    if len(files) != 1:
        raise ValueError('Require one real first query')
    with np.load(files[0], allow_pickle=False) as z:
        return {k: z[k].copy() for k in ['image', 'wrist_image', 'state', 'instruction']}


def existing_candidate(root, case, checkpoint_sha):
    paths = set((root / 'runs').glob('*/result.json'))
    for pattern in ['*/case-runs/*/result.json', '*/stages/*/case-runs/*/result.json']:
        paths.update((root / 'control').glob(pattern))
    for path in paths:
        z = read_json(path)
        if (z.get('checkpoint_receipt_sha256') == checkpoint_sha and
                z.get('condition') == case['condition'] and z.get('suite') == case['suite'] and
                z.get('task_id') == case['task_id'] and z.get('initial_state_index') == case['reset_index']):
            return path
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--round-dir', type=Path, required=True)
    p.add_argument('--suite', required=True)
    p.add_argument('--condition', choices=CONDITIONS, required=True)
    a = p.parse_args()
    root, rd = a.root.resolve(), a.round_dir.resolve()
    series, config = read_json(rd / 'series.json'), read_json(rd / 'config.json')
    validate_manifest(series)
    auth = read_json(rd / 'authorization.json')
    # Reject CPU templates before even asking for GPU resources.
    if not series.get('executable') or not auth.get('gpu_execution_authorized'):
        raise ValueError('Only a future explicitly authorized GPU plan may execute')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    require_current_authorization(auth, series, rd, config, socket.gethostname(), gpu)
    if not series.get('root_control'):
        raise ValueError('Register a fresh current-round shutdown guard before GPU work')
    shutdown_root = Path(series['root_control'])
    parent_auth = read_json(shutdown_root / 'authorization.json')
    if (parent_auth.get('shutdown_authorized') is not True or parent_auth.get('gpu_uuid') != gpu or
            parent_auth.get('hostname') != socket.gethostname()):
        raise ValueError('Current shutdown authority differs from this GPU round')
    guard = read_json(shutdown_root / 'guard-launch-receipt.json')
    guard_cmd = Path('/proc', str(guard['pid']), 'cmdline')
    if not guard_cmd.exists() or str(shutdown_root).encode() not in guard_cmd.read_bytes():
        raise ValueError('Current root shutdown guard is not actually running')
    for path, expected in series['frozen_files'].items():
        if digest(Path(path)) != expected:
            raise ValueError('Frozen source/profile/reference changed: ' + path)
    if digest(root / 'models/openvla_oft/DOWNLOAD_VERIFICATION.json') != series['checkpoint_receipt_sha256']:
        raise ValueError('Checkpoint receipt changed')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', MUJOCO_GL='egl', OMP_NUM_THREADS='4',
               PYTHONPATH=os.pathsep.join(map(str, [root / 'code-cpu-prep-20261109-v1',
                                                    root / 'sources/openvla-oft', root / 'sources/LIBERO'])))
    progress = rd / (a.condition + '-' + a.suite + '-cases.json')
    state = dict(state='RUNNING', condition=a.condition, suite=a.suite, completed=[])
    for name in ['case-runs', 'case-logs']:
        (rd / name).mkdir(exist_ok=True)
    for case in series['cases']:
        if case['condition'] != a.condition or case['suite'] != a.suite:
            continue
        if (rd / 'STOP_NEW_GPU').exists() or quota_decision(read_json(rd / 'quota.json'), config['control']) != 'CONTINUE':
            state.update(state='STOPPED_NO_NEW_CASES', updated_at=now())
            write_json(progress, state)
            return
        if shutil.disk_usage(root).free < 6 * 1024**3:
            raise ValueError('Runtime disk reserve below6GiB')
        if subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip() != gpu:
            raise ValueError('GPU identity changed')
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            raise ValueError('Another GPU process is active')
        if existing_candidate(root, case, series['checkpoint_receipt_sha256']):
            raise ValueError('Already existing candidate requires audit/reindex; never rerun')
        reference = Path(case['reference_result'])
        if digest(reference) != case['reference_sha256']:
            raise ValueError('Reference changed')
        case_id = f"{rd.name}-{a.condition}-{a.suite}-task{case['task_id']}-reset25"
        output = rd / 'case-runs' / case_id
        if output.exists():
            raise FileExistsError('Never overwrite any partial or complete output')
        command = build_command(root, series, case, output, case_id)
        state.update(current_case=case, argv=command, updated_at=now())
        write_json(progress, state)
        started = time.monotonic()
        with (rd / 'case-logs' / (case_id + '.log')).open('xb') as log:
            child = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            state['policy_pid'] = child.pid
            write_json(progress, state)
            code = child.wait()
        if code:
            state.update(state='CASE_PROCESS_FAILED', exit_code=code)
            write_json(progress, state)
            raise RuntimeError('Collector failed; preserve partial output')
        z = read_json(output / 'result.json')
        if (not z['gate'].startswith('PASS_ACTUAL_OFT_') or z['condition'] != a.condition or
                z['checkpoint_receipt_sha256'] != series['checkpoint_receipt_sha256'] or
                z['profile_sha256'] != series['profiles'][a.condition]['sha256'] or
                z['initial_state_sha256'] != case['initial_state_sha256']):
            raise ValueError('Actual result identity differs')
        left, right = first_input(reference.parent), first_input(output)
        if any(not np.array_equal(left[k], right[k]) for k in left):
            raise ValueError('Actual first input differs from exact BF16 reference')
        state['completed'].append(dict(case=case, result_path=str(output / 'result.json'),
                                       result_sha256=digest(output / 'result.json'),
                                       wall_seconds=time.monotonic() - started, first_input_paired=True))
        write_json(progress, state)
    state.update(state='COMPLETE_PENDING_FULL_RAW_AUDIT', updated_at=now())
    write_json(progress, state)


if __name__ == '__main__':
    main()
