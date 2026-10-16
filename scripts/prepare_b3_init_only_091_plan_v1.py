"""Register the fixed PTQ/S0 mechanism controls; registration never starts GPU."""
from __future__ import annotations
import argparse
import json
import socket
import sys
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qvla.extended_peft import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('control', 'materials', 'session', 'cpu-gate', 'observation-gate',
                 'reclaim-gate', 'source-manifest'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--expected-hostname', required=True)
    p.add_argument('--preregistration-commit', required=True)
    p.add_argument('--include-training80-diagnostic', action='store_true')
    a = p.parse_args()
    if socket.gethostname() != a.expected_hostname:
        raise ValueError('Wrong current instance')
    if (a.control / 'STOP_NEW_GPU').exists():
        raise ValueError('Client requested no new GPU work')
    if shutil.disk_usage(a.control).free < 12 * 1024**3:
        raise ValueError('Need12GiB persistent disk before new GPU candidate')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
            '--format=csv,noheader'], text=True).strip():
        raise ValueError('GPU is occupied; never duplicate a running experiment')
    checks = ((a.cpu_gate, 'PASS_B3_INIT_ONLY_CPU_METADATA_CONTRACTS'),
              (a.observation_gate, 'PASS_CPU_PROTOCOL_OBSERVATION_BRIDGE'),
              (a.reclaim_gate, 'PASS_SVD_SPOOL_RECLAIM_BACKED_BY_LOCAL_COPY'))
    for path, expected in checks:
        if json.loads(path.read_text())['gate'] != expected:
            raise ValueError('Missing prerequisite: ' + expected)
    m = json.loads(a.materials.read_text())
    old = Path('/root/autodl-tmp/qvla-repro/backups/experiments/p2-shared-peft/'
               '20261015-046-quant-finetuning-v1/response-svd/steps-1000')
    parent, initial = old / 'artifact.json', old / 'initial-adapter.pt'
    r = json.loads(parent.read_text())
    if sha(initial) != r['reuse_files']['initial-adapter.pt']:
        raise ValueError('Training-preceding factors do not match completed S artifact')
    files = {}
    source = json.loads(a.source_manifest.read_text())
    for name, expected in source.items():
        path = ROOT / name
        if sha(path) != expected:
            raise ValueError('Frozen source changed: ' + name)
        files[str(path)] = expected
    for path in (a.materials, a.cpu_gate, a.observation_gate, a.reclaim_gate,
                 a.source_manifest, parent, initial,
                 Path(m['student80']) / 'manifest.json',
                 Path(m['oft_root']) / 'experiments/robot/libero/run_libero_eval.py',
                 Path(m['oft_root']) / 'experiments/robot/openvla_utils.py',
                 *(Path(m[k]) for k in ('w2_g128', 'w2_g64'))):
        files[str(path)] = sha(path)
    output = a.control / 'M0-plans-v1'
    output.mkdir(exist_ok=False)
    lock = output / 'runtime-lock.json'
    lock.write_text(json.dumps({'version': 'b3-init-only-091-v1-20261016',
        'expected_hostname': a.expected_hostname, 'files': files,
        'max_stage_seconds': 10800, 'max_plan_seconds': 21600}, indent=2) + '\n')
    env = {'PYTHONPATH': ':'.join((str(ROOT), str(ROOT / 'diagnostics'),
            m['oft_root'], m['libero_root'])), 'CUDA_VISIBLE_DEVICES': '0',
        'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl', 'MUJOCO_EGL_DEVICE_ID': '0',
        'PYTHONHASHSEED': '7', 'WANDB_MODE': 'disabled', 'TOKENIZERS_PARALLELISM': 'false',
        'TF_NUM_INTRAOP_THREADS': '1', 'TF_NUM_INTEROP_THREADS': '1'}
    py = '/root/miniconda3/envs/qvla-oft/bin/python'
    stages = []
    def add(name, command, directory):
        stages.append({'id': name, 'cwd': str(ROOT), 'env': env,
            'command': list(map(str, [py, ROOT / 'scripts/run_locked_peft_stage.py',
                '--lock', lock, '--output', directory, '--', *command]))})
    if a.include_training80_diagnostic:
        add('training80-checkpoint-diagnostic',
            [py, ROOT / 'scripts/diagnose_b3_checkpoints_091_v1.py',
             '--materials', a.materials,
             '--old-session', old.parent.parent,
             '--output', a.session / 'diagnostics' / 'training80'],
            a.session / 'stage-records' / 'training80-checkpoint-diagnostic')
    for phase, count in (('micro', 1), ('first50', 5)):
        for candidate, model in (('ptq', 'A4'), ('svd-init-only', 'B3-S0-091-v1')):
            directory = a.session / phase / candidate
            command = [py, ROOT / 'qvla/run_eval_b3_init_only_091_v1.py',
                '--candidate', candidate, '--model-id', model,
                '--method', 'awq', '--weight-bits', '2', '--activation-bits', '16',
                '--profile', m['w2_g128'], '--awq-primary-group64-profile', m['w2_g64'],
                '--awq-candidate', 'w2-attention-no-clip-primary-g64',
                '--official-root', m['official_root'], '--pretrained_checkpoint', m['checkpoint'],
                '--libero_root', m['libero_root'], '--task_suite_name', 'libero_spatial',
                '--initial-state-offset', '5', '--num_trials_per_task', str(count),
                '--seed', '0', '--env-seed', '1', '--seed-protocol', 'paired',
                '--trace-actions', '--trace-observations', '--local_log_dir', directory,
                '--initial-artifact', parent, '--initial-artifact-sha256', sha(parent),
                '--initial-state', initial, '--recovery-training-manifest',
                Path(m['student80']) / 'manifest.json']
            add(phase + '-' + candidate, command, directory)
        add('audit-' + phase, [py, ROOT / 'scripts/audit_b3_init_only_091_v1.py',
            '--root', a.session / phase, '--offset', '5', '--count', str(count),
            '--output', a.session / (phase + '-gate.json')],
            a.session / 'stage-records' / ('audit-' + phase))
    plan = {'schema_version': '1.0', 'plan_id': a.session.name + '-M0-first50',
        'control_root': str(output / 'control'), 'terminal_state': 'AWAITING_GATE_REVIEW',
        'stages': stages, 'preregistration_commit': a.preregistration_commit,
        'question': 'Fixed A4 PTQ versus the original pre-training SVD factors; no optimizer steps',
        'formal_registered_total_per_candidate': 450, 'this_plan_formal_per_candidate': 50,
        'micro_overlap_excluded': True,
        'next': 'Review strict gate, paired failures/cost, live primary quota/disk; '
                'continue registered reset10-19/20-29/30-39/40-49 without score-direction stopping'}
    from scripts.vla_stage_supervisor import validate_plan
    validate_plan(plan)
    path = output / 'first50-plan.json'
    path.write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({'gate': 'PASS_B3_INIT_ONLY_PLAN_REGISTERED', 'plan': str(path),
                      'file_sha256': sha(path), 'stages': len(stages), 'gpu_started': False}))


if __name__ == '__main__':
    main()
