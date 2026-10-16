"""Register the versioned, six-stage temporal comparison; registration never starts GPU."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--preregistration-commit', required=True)
    p.add_argument('--primary-remaining', type=int, required=True)
    p.add_argument('--primary-observed-at', required=True)
    p.add_argument('--primary-resets-at', type=int, required=True)
    a = p.parse_args()
    c = a.control
    b = c / 'B1-temporal-v1'
    r = Path('/root/autodl-tmp/qvla-repro/backups/experiments/p2-shared-peft/20261016-091-residual-resume-v2')
    src = c.parent / '20261016-091-residual-vla-adaptation-v1/source-v1'
    now = datetime.now(timezone.utc)
    if (not 0 <= (now - datetime.fromisoformat(a.primary_observed_at)).total_seconds() <= 300
            or a.primary_resets_at <= now.timestamp() or a.primary_remaining <= 15
            or (c / 'STOP_NEW_GPU').exists()):
        raise ValueError('Fresh primary >15 and no stop marker required for new registration')
    if socket.gethostname() != 'autodl-container-dfbc4bb0ff-1e85dd94' or shutil.disk_usage(c).free < 12 * 1024**3:
        raise ValueError('Wrong host or <12GiB new-method admission')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise ValueError('GPU occupied')
    cpu = json.loads((b / 'cpu-gate.json').read_text())
    if cpu['gate'] != 'PASS_B1_TEMPORAL_CPU_CONTRACTS' or not cpu['variant_substitution_rejected']:
        raise ValueError('Missing temporal CPU gate')
    for f, expected in cpu['source_sha256'].items():
        if sha(f) != expected:
            raise ValueError('Changed CPU-admitted source/material')
    for phase, gate, count in [('first50-recovered', 'PASS_B3_INIT_ONLY_SEGMENTED_PAIRED_PROTOCOL', 50)] + [
            ('reset-%d-%d' % (i,i+9), 'PASS_B3_INIT_ONLY_FULL_SHARD_PAIRED_PROTOCOL', 100) for i in (10,20,30,40)]:
        g = json.loads((r / (phase + '-gate.json')).read_text())
        if g['gate'] != gate or g['paired_episodes'] != count:
            raise ValueError('M0 not fully admitted')
    sys.path.insert(0, str(src))
    from scripts.vla_stage_supervisor import validate_plan
    lock = json.loads((c / 'M0-continuation-v2/reset-40-49/runtime-lock.json').read_text())
    files = lock['files']
    for f in [*b.glob('*.py'), *b.glob('*.json'), r / 'reset-40-49-gate.json',
              c / 'M0_450_ANALYSIS_CN.md', c / 'B1_TEMPORAL_PRE_RUN_CARD_CN.md']:
        files[str(f)] = sha(f)
    for v in ('early80', 'phase80'):
        data = r / 'B1-temporal-v1/data' / v
        for f in data.glob('*'):
            if f.is_file():
                files[str(f)] = sha(f)
    gatepath = r / 'B1-temporal-v1/data/data-gate.json'
    files[str(gatepath)] = sha(gatepath)
    lock.update(version='b1-temporal80-v1-20261017', max_stage_seconds=10800, max_plan_seconds=21600)
    planroot = b / 'training-plan-v1'
    planroot.mkdir(exist_ok=False)
    lockpath = planroot / 'runtime-lock.json'
    lockpath.write_text(json.dumps(lock, indent=2) + '\n')
    py = '/root/miniconda3/envs/qvla-oft/bin/python'
    env = {'PYTHONPATH': ':'.join((str(src), str(src / 'diagnostics'), str(b),
           '/root/autodl-tmp/qvla-repro/control/cpu-audit-20261012/oft-corrected-v1',
           '/root/autodl-tmp/qvla-repro/src/LIBERO')), 'VLA_FROZEN_SOURCE': str(src),
           'CUDA_VISIBLE_DEVICES': '0', 'MUJOCO_GL': 'egl', 'MUJOCO_EGL_DEVICE_ID': '0',
           'PYOPENGL_PLATFORM': 'egl', 'PYTHONHASHSEED': '7', 'WANDB_MODE': 'disabled',
           'TOKENIZERS_PARALLELISM': 'false', 'TF_NUM_INTEROP_THREADS': '1', 'TF_NUM_INTRAOP_THREADS': '1'}
    stages = []
    out = r / 'B1-temporal-v1'
    def add(name, command):
        stages.append({'id': name, 'cwd': str(src), 'env': env,
                       'command': [py, str(src / 'scripts/run_locked_peft_stage.py'), '--lock', str(lockpath),
                                   '--output', str(out / 'stage-records' / name), '--', *command]})
    for steps in (10, 1000):
        for v in ('early80', 'phase80'):
            cmd = [py, str(b / 'train_b1_temporal80_091_v1.py'), '--model-id', 'B3',
                   '--initialization', 'response-svd', '--materials', str(b / (v + '-materials.json')),
                   '--steps', str(steps), '--output', str(out / v / ('steps-%d' % steps))]
            if steps == 1000:
                cmd += ['--smoke-dir', str(out / v / 'steps-10')]
            add('train-%d-%s' % (steps,v), cmd)
        add('audit-training-%d' % steps, [py, str(b / 'audit_b1_temporal_train_091_v1.py'),
            '--root', str(out), '--steps', str(steps), '--output', str(out / ('train-%d-pair-gate.json' % steps))])
    plan = {'schema_version': '1.0', 'plan_id': '20261017-091-B1-temporal80-training-v1',
            'control_root': str(planroot / 'control'), 'terminal_state': 'AWAITING_GATE_REVIEW',
            'stages': stages, 'preregistration_commit': a.preregistration_commit,
            'single_variable': 'query0/1 vs round(.25/.75*(queries-1)) in same40 training rollouts',
            'same_initial_SVD_factors': True, 'same_optimizer_loss_steps': True,
            'formal_episodes_in_this_plan': 0, 'primary_remaining_at_registration': a.primary_remaining,
            'primary_observed_at': a.primary_observed_at, 'primary_resets_at': a.primary_resets_at,
            'next': 'Training analysis and full backups; independently versioned evaluation loader/numerical bridge/micro/first50 gate before closed-loop expansion'}
    validate_plan(plan)
    path = planroot / 'plan.json'
    path.write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({'gate': 'PASS_B1_TEMPORAL_PLAN_REGISTERED', 'plan': str(path),
                      'sha256': sha(path), 'stages': len(stages), 'gpu_started': False}))


if __name__ == '__main__':
    main()
