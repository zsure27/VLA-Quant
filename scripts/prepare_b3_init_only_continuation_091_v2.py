"""Register next M0 shard in fixed order; only offset/count/output/lock change."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--offset', type=int, choices=(10, 20, 30, 40), required=True)
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--preregistration-commit', required=True)
    p.add_argument('--primary-remaining', type=int, required=True)
    p.add_argument('--primary-observed-at', required=True)
    p.add_argument('--primary-resets-at', type=int, required=True)
    a = p.parse_args()
    c = a.control
    observed = datetime.fromisoformat(a.primary_observed_at)
    now = datetime.now(timezone.utc)
    if (not 0 <= (now - observed).total_seconds() <= 300 or a.primary_resets_at <= now.timestamp()
            or a.primary_remaining <= 15 or (c / 'STOP_NEW_GPU').exists()):
        raise ValueError('Need fresh current primary quota above15%; no stop marker')
    if shutil.disk_usage(c).free < 8 * 1024**3:
        raise ValueError('Need8GiB for next shard: reserve6GiB plus2GiB')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise ValueError('GPU occupied')
    source = c.parent / '20261016-091-residual-vla-adaptation-v1/source-v1'
    sys.path.insert(0, str(source))
    from scripts.vla_stage_supervisor import validate_plan
    old = source.parent
    session = Path('/root/autodl-tmp/qvla-repro/backups/experiments/p2-shared-peft/20261016-091-residual-resume-v2')
    previous = session / ('first50-recovered-gate.json' if a.offset == 10 else 'reset-%d-%d-gate.json' % (a.offset - 10, a.offset - 1))
    gate = json.loads(previous.read_text())
    expected_gate = 'PASS_B3_INIT_ONLY_SEGMENTED_PAIRED_PROTOCOL' if a.offset == 10 else 'PASS_B3_INIT_ONLY_FULL_SHARD_PAIRED_PROTOCOL'
    if gate['gate'] != expected_gate or gate['paired_episodes'] != (50 if a.offset == 10 else 100):
        raise ValueError('Missing completed previous strict gate')
    if not a.analysis.is_file() or a.analysis.stat().st_size < 100:
        raise ValueError('Previous phase analysis must exist before continuation')
    baseline = json.loads((old / 'M0-plans-v1/first50-plan.json').read_text())
    original_lock = json.loads((old / 'M0-plans-v1/runtime-lock.json').read_text())
    lock = copy.deepcopy(original_lock)
    phase = 'reset-%d-%d' % (a.offset, a.offset + 9)
    out = c / 'M0-continuation-v2' / phase
    out.mkdir(parents=True, exist_ok=False)
    lockpath = out / 'runtime-lock.json'
    for f in (previous, a.analysis, c / 'audit_b3_complete_phase_091_v2.py',
              c / 'audit_b3_segments_091_v3.py', c / 'prepare_b3_init_only_continuation_091_v2.py', c / 'runtime-admission.json'):
        lock['files'][str(f)] = sha(f)
    lock['version'] = '091-init-only-continuation-v2-' + phase
    lockpath.write_text(json.dumps(lock, indent=2) + '\n')
    stages = []
    for v in ('ptq', 'svd-init-only'):
        s = copy.deepcopy(next(x for x in baseline['stages'] if x['id'] == 'first50-' + v))
        s['id'] = phase + '-' + v
        cmd = s['command']
        cmd[cmd.index('--lock') + 1] = str(lockpath)
        cmd[cmd.index('--output') + 1] = str(session / phase / v)
        for flag, value in (('--initial-state-offset', str(a.offset)), ('--num_trials_per_task', '10'),
                            ('--local_log_dir', str(session / phase / v))):
            cmd[cmd.index(flag) + 1] = value
        stages.append(s)
    env = dict(stages[0]['env'])
    py = stages[0]['command'][0]
    stages.append({'id': 'audit-' + phase, 'cwd': str(source), 'env': env,
                   'command': [py, str(source / 'scripts/run_locked_peft_stage.py'), '--lock', str(lockpath),
                   '--output', str(session / 'stage-records' / ('audit-' + phase)), '--', py,
                   str(c / 'audit_b3_complete_phase_091_v2.py'), '--control', str(c), '--frozen-source', str(source),
                   '--root', str(session / phase), '--offset', str(a.offset), '--output', str(session / (phase + '-gate.json'))]})
    plan = {'schema_version': '1.0', 'plan_id': '20261016-091-residual-resume-v2-' + phase,
            'control_root': str(out / 'control'), 'terminal_state': 'AWAITING_GATE_REVIEW',
            'stages': stages, 'preregistration_commit': a.preregistration_commit,
            'previous_gate_sha256': sha(previous), 'previous_analysis_sha256': sha(a.analysis),
            'primary_remaining_at_registration': a.primary_remaining,
            'primary_observed_at': a.primary_observed_at, 'primary_resets_at': a.primary_resets_at,
            'formal_per_candidate_this_shard': 100, 'formal_total_per_candidate': 450,
            'GPU_fields_changed': ['offset', 'count', 'output', 'lock'], 'score_direction_used_for_continuation': False}
    validate_plan(plan)
    planpath = out / 'plan.json'
    planpath.write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({'gate': 'PASS_M0_CONTINUATION_REGISTERED', 'plan': str(planpath),
                      'sha256': sha(planpath), 'gpu_started': False}))


if __name__ == '__main__':
    main()
