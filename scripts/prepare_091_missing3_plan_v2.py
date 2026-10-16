"""Register only reboot admission, missing three S0 episodes, and segmented hard gate."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--preregistration-commit', required=True)
    a = p.parse_args()
    old = Path('/root/autodl-tmp/qvla-repro/control/20261016-091-residual-vla-adaptation-v1')
    source = old / 'source-v1'
    sys.path.insert(0, str(source))
    from scripts.vla_stage_supervisor import validate_plan
    c = a.control
    if (c / 'STOP_NEW_GPU').exists():
        raise ValueError('No new GPU requested')
    original = json.loads((old / 'M0-plans-v1/first50-plan.json').read_text())
    prior_stage = next(s for s in original['stages'] if s['id'] == 'first50-svd-init-only')
    command = prior_stage['command']
    frozen = json.loads((c / 'recovery-preflight.json').read_text())
    if frozen['gate'] != 'PASS_M0_47_PLUS_3_RECOVERY_PREFLIGHT':
        raise ValueError('No exact completed prefix proof')
    helpers = ('audit_b3_segmented_091_v2.py', 'eval_b3_missing3_091_v2.py',
               'b3_episode_resume_v2.py', 'verify_091_resume_admission_v2.py',
               'prepare_091_missing3_plan_v2.py', 'recovery-preflight.json',
               'windows-shutdown-authorization-v2.json')
    lock = {'version': '091-missing3-resume-v2', 'expected_hostname': original.get('expected_hostname', 'autodl-container-dfbc4bb0ff-1e85dd94'),
            'files': dict(frozen['frozen_files']), 'max_stage_seconds': 3600, 'max_plan_seconds': 5400}
    lock['files'].update({str(c / n): sha(c / n) for n in helpers})
    directory = c / 'recovery-plan-v2'
    directory.mkdir(exist_ok=False)
    lock_path = directory / 'runtime-lock.json'
    lock_path.write_text(json.dumps(lock, indent=2) + '\n')
    session = Path('/root/autodl-tmp/qvla-repro/backups/experiments/p2-shared-peft/20261016-091-residual-resume-v2')
    old_session = session.parent / '20261016-091-residual-vla-adaptation-v1'
    resumed = session / 'missing3/svd-init-only'
    py = command[0]
    env = dict(prior_stage['env'])
    stages = []
    def stage(name, scientific, output):
        stages.append({'id': name, 'cwd': str(source), 'env': env,
                       'command': [py, str(source / 'scripts/run_locked_peft_stage.py'),
                                   '--lock', str(lock_path), '--output', str(output), '--', *map(str, scientific)]})
    pre_sha = sha(c / 'recovery-preflight.json')
    stage('reboot-and-recovery-admission', [py, c / 'verify_091_resume_admission_v2.py',
          '--control', c, '--old-control', old, '--preflight-sha256', pre_sha,
          '--output', c / 'runtime-admission.json'], session / 'stage-records/recovery-admission')
    argv = copy.copy(command[command.index('--') + 1:])
    argv[1] = str(c / 'eval_b3_missing3_091_v2.py')
    argv[2:2] = ['--frozen-source', str(source), '--resume-preflight', str(c / 'recovery-preflight.json'),
                 '--resume-preflight-sha256', pre_sha, '--schedule-module', str(c / 'b3_episode_resume_v2.py')]
    for flag, value in (('--initial-state-offset', '7'), ('--num_trials_per_task', '3'), ('--local_log_dir', str(resumed))):
        argv[argv.index(flag) + 1] = value
    stage('recover-only-task9-reset7-9', argv, resumed)
    stage('audit-recovered-first50', [py, c / 'audit_b3_segmented_091_v2.py',
          '--frozen-source', source, '--old-control', old, '--old-session', old_session,
          '--resumed', resumed, '--preflight', c / 'recovery-preflight.json',
          '--output', session / 'first50-recovered-gate.json'], session / 'stage-records/audit-recovered-first50')
    plan = {'schema_version': '1.0', 'plan_id': '20261016-091-residual-resume-v2-recover-first50',
            'control_root': str(directory / 'control'), 'terminal_state': 'AWAITING_GATE_REVIEW',
            'stages': stages, 'preregistration_commit': a.preregistration_commit,
            'formal_registered_per_candidate': 450, 'additional_GPU_episodes': 3,
            'old_complete_episodes_repeated': False,
            'next': 'Analyze complete paired50/failures/cost, fresh primary quota/disk, then registered reset10-19',
            'interrupted_original_prefix_and_attempt_preserved': True}
    validate_plan(plan)
    path = directory / 'plan.json'
    path.write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({'gate': 'PASS_MISSING3_PLAN_REGISTERED', 'plan': str(path),
                      'sha256': sha(path), 'preflight_sha256': pre_sha, 'gpu_started': False}))


if __name__ == '__main__':
    main()
