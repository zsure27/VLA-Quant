"""Register or audit fixed AWQ cases after the actual full-policy gate.

Registration is CPU only. Launch remains an explicit controller action, never a
server-side experiment chooser. Pilot cases count once in the formal first 40.
"""
import argparse
import datetime
import json
from pathlib import Path
import socket
import sys

from qvla.infra.common import canonical, digest, now, write_json
from qvla.infra.plan import compile_plan, verify_plan
from qvla.pi05_torch_backend_v2 import BACKEND_ID
from qvla.audit_pi05_torch_trace_v2 import audit as audit_bf16
from qvla.audit_pi05_torch_awq_trace_v1 import audit as audit_awq

SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
GATE = 'PASS_AWQ_REGISTERED_CASES_FULL_RAW_AUDIT'


def load(path):
    return json.loads(Path(path).read_text())


def audit_stage(stage):
    stage = Path(stage)
    series = load(stage / 'series.json')
    rows = []
    for case in series['cases']:
        name = case['round_id']
        plan = stage / 'case-plans' / (name + '-runtime.json')
        template = stage / 'case-plans' / (name + '.json')
        if digest(template) != case['template_sha256']:
            raise ValueError('Frozen case template changed')
        actual = load(plan)
        if actual['profile_sha256'] != series['profile_sha256']:
            raise ValueError('Case profile differs')
        output = Path(actual['output'])
        proof = audit_awq(output, plan)
        result = load(output / 'result.json')
        rows.append(dict(suite=case['suite'], task_id=case['task_id'],
                         reset_index=case['reset_index'], round_id=name,
                         success=bool(result['success']), audit=proof,
                         result_path=str(output / 'result.json'),
                         result_sha256=digest(output / 'result.json')))
    expected = {(s, t, series.get('reset_index',5)) for s in SUITES for t in series['task_ids']}
    keys = {(x['suite'], x['task_id'], x['reset_index']) for x in rows}
    if len(rows) != len(expected) or keys != expected:
        raise ValueError('Incomplete or duplicate predeclared AWQ cases')
    report = dict(gate=GATE, scope=series['scope'], backend=BACKEND_ID,
                  profile_sha256=series['profile_sha256'], reset_index=series.get('reset_index',5), complete=len(rows),
                  successes=sum(x['success'] for x in rows), rows=rows,
                  scores_not_used_for_continuation=True, observed_at=now())
    destination = stage / 'FULL_RAW_AUDIT.json'
    if destination.exists():
        prior = load(destination)
        for key in ('gate', 'scope', 'profile_sha256', 'complete', 'rows'):
            if prior[key] != report[key]:
                raise ValueError('Existing immutable audit differs')
        return prior
    write_json(destination, report)
    return report


def prepare(args):
    root, control, code = args.root.resolve(), args.control.resolve(), args.code.resolve()
    calibration = args.calibration_stage.resolve()
    state = load(calibration / 'status.json')
    if state['state'] != 'PAUSED_SERVER_LEFT_ON' or any(x.get('exit_code') != 0 for x in state['stages']):
        raise ValueError('Calibration stage must actually finish successfully')
    auth = load(control / 'authorization.json')
    if auth['hostname'] != socket.gethostname() or auth.get('primary_only') is not True:
        raise ValueError('Current round identity differs')
    if datetime.datetime.fromisoformat(auth['expires_at']) <= datetime.datetime.now(datetime.timezone.utc):
        raise ValueError('Current authorization expired')
    profile = calibration / 'profile/profile.pt'
    numerical = profile.with_name('result.json')
    if load(numerical)['gate'] != 'PASS_PI05_AWQ_NATIVE_414_RELOAD_FINITE':
        raise ValueError('Full-policy numerical gate not passed')
    from qvla.pi05_awq_episode_admission_v1 import prepare as prepare_probes, load_probe
    probes = profile.with_name('episode-admission.json')
    if not probes.exists():
        prepare_probes(profile, root / 'rounds/20261111-084-pi05-smoothquant-a8-v2/capture', probes)
    config = load(calibration / 'config.json')
    load_probe(probes, profile, config['model']['identity_sha256'])
    tasks = [0] if args.scope == 'pilot4' else list(range(1, 10))
    if args.scope == 'remaining36':
        if not args.pilot_stage:
            raise ValueError('Four-suite pilot audit is required')
        pilot = audit_stage(args.pilot_stage)
        if (pilot['scope'] != 'pilot4' or pilot['complete'] != 4 or
                pilot['profile_sha256'] != digest(profile) or pilot.get('reset_index',5)!=args.reset_index):
            raise ValueError('Pilot does not use this actual profile')
    paths = [*(root / 'rounds').glob('*/case-runs/*/result.json'),
             *(root / 'control').glob('*/stages/*/case-runs/*/result.json')]
    refs = {}
    existing = set()
    for path in paths:
        result = load(path)
        if result.get('backend') != BACKEND_ID or result.get('role') != 'development':
            continue
        key = (result.get('suite'), result.get('task_id'), result.get('reset_index'))
        if key not in {(s, t, args.reset_index) for s in SUITES for t in tasks}:
            continue
        if result.get('condition') == 'awq_w4a16' and result.get('gate') == 'PASS_PI05_TORCH_AWQ_EPISODE':
            existing.add(key)
        if result.get('condition') == 'bf16' and result.get('gate') == 'PASS_PI05_TORCH_BF16_EPISODE':
            if key in refs:
                raise ValueError('Duplicate BF16 reference')
            refs[key] = path
    if existing:
        raise ValueError('Requested scope already includes completed AWQ cases; do not repeat')
    if len(refs) != 4 * len(tasks):
        raise ValueError('Missing same-backend same-case BF16 reference')
    selected = []
    for suite in SUITES:
        for task in tasks:
            reference = refs[suite, task, args.reset_index]
            original = load(reference)
            folder = reference.parent.parent.parent / 'case-plans'
            candidates = [folder / (original['round_id'] + suffix) for suffix in ('-runtime.json', '.json')]
            plans = [p for p in candidates if p.is_file() and digest(p) == original['plan_sha256']]
            if len(plans) != 1:
                raise ValueError('Reference execution plan is unavailable or ambiguous')
            audit_bf16(reference.parent, plans[0])
            selected.append((suite, task, reference, load(plans[0])))
    stage = control / 'stages' / args.round_id
    stage.mkdir(exist_ok=False)
    for name in ('case-plans', 'case-runs', 'case-logs', 'audits'):
        (stage / name).mkdir()
    cases, inputs = [], []
    for suite, task, reference, template in selected:
        name = f'{args.round_id}-{suite}-task{task}-reset{args.reset_index}'
        template.update(round_id=name, server_id='084', hostname=auth['hostname'],
                        gpu_uuid=auth['gpu_uuid'], runner='qvla.execute_pi05_torch_awq_episode_v1',
                        condition='awq_w4a16', role='development', suite=suite,
                        task_id=task, reset_index=args.reset_index, model_seed=7, backend=BACKEND_ID,
                        bf16_reference=str(reference.parent), model_identity_sha256=config['model']['identity_sha256'],
                        output=str(stage / 'case-runs' / name), registered_at=now())
        path = stage / 'case-plans' / (name + '.json')
        write_json(path, template)
        cases.append(dict(round_id=name, suite=suite, task_id=task, reset_index=args.reset_index,
                          role='development', template_sha256=digest(path)))
        inputs += [dict(path=str(path), sha256=digest(path)),
                   dict(path=str(reference), sha256=digest(reference)),
                   dict(path=str(reference.parent / 'query-00000.npz'), sha256=digest(reference.parent / 'query-00000.npz'))]
    admission = Path(selected[0][3]['admission_path'])
    if any(Path(x[3]['admission_path']) != admission for x in selected):
        raise ValueError('Independent BF16 admissions differ')
    config['task']['changed_variable'] = 'Original AWQ W4A16, same backend/seed/noise/cadence; no fine-tuning'
    identity_file = code / 'Ex_Reports/_tasks' / config['task']['id'] / 'task.json'
    if identity_file.exists():
        config['task'] = load(identity_file)
    config['quantization'] = dict(method='awq', weight_bits=4, activation_bits=16,
                                  profiles=[dict(path=str(profile), sha256=digest(profile))])
    config['control'].update(start_disk_gib=8, refresh_confirmation_wait_seconds=120)
    config['control'].pop('weekly_close_remaining_percent', None)
    for suite in config['evaluation']['suites']:
        suite.update(task_ids=tasks, reset_offset=args.reset_index, resets_per_task=1)
    binding = config['binding']
    for name in ('admission_argv', 'post_evaluation_argv', 'post_training_audit_argv', 'evaluation_audit_argv'):
        binding.pop(name, None)
    binding.update(id='pi05-awq-native-formal-first40-v1', supports_task_subsets=True,
                   evaluation_argv=['{python}', '{repo}/scripts/trajectory_drift_20261116/run_pi05_awq_cases_v1.py',
                                    '--root', str(root), '--round-dir', str(stage), '--suite', '{suite}',
                                    '--role', 'development', '--activation-bits', '16'])
    binding['gate_files'] += [dict(path=str(numerical), sha256=digest(numerical), expected_gate=load(numerical)['gate']),
                              dict(path=str(probes), sha256=digest(probes), expected_gate=load(probes)['gate'])]
    if args.pilot_stage:
        proof = args.pilot_stage / 'FULL_RAW_AUDIT.json'
        binding['gate_files'].append(dict(path=str(proof), sha256=digest(proof), expected_gate=GATE))
    write_json(stage / 'config.json', config)
    series = dict(cases=cases, task_ids=tasks, scope=args.scope, reset_index=args.reset_index, control=config['control'],
                  bf16_admission=str(admission), profile=str(profile), profile_sha256=digest(profile),
                  pilot_counts_once=True, performance_not_a_continuation_gate=True)
    write_json(stage / 'series.json', series)
    plan = compile_plan(config, code, stage)
    for rel in ['qvla/pi05_awq_profile_v1.py', 'qvla/pi05_awq_episode_admission_v1.py',
                'qvla/pi05_torch_backend_v2.py', 'qvla/execute_pi05_torch_awq_episode_v1.py',
                'qvla/audit_pi05_torch_awq_trace_v1.py', 'scripts/trajectory_drift_20261116/run_pi05_awq_cases_v1.py',
                'scripts/trajectory_drift_20261118/prepare_pi05_awq_cases_v2.py']:
        plan['source_sha256'][str(code / rel)] = digest(code / rel)
    plan['inputs'] += inputs + [dict(path=str(stage / 'series.json'), sha256=digest(stage / 'series.json')),
                                dict(path=str(admission), sha256=digest(admission))]
    plan.pop('plan_sha256')
    plan['plan_sha256'] = canonical(plan)
    verify_plan(plan)
    write_json(stage / 'plan.json', plan)
    write_json(stage / 'CPU_PLAN_READY.json', dict(gate='PASS_AWQ_ACTUAL_PROFILE_AND_REFERENCES_CPU_REGISTRATION',
               plan_sha256=plan['plan_sha256'], cases=len(cases), gpu_not_started=True))
    return dict(stage=str(stage), plan_sha256=plan['plan_sha256'], cases=len(cases), gpu_not_started=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-stage', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--control', type=Path)
    parser.add_argument('--code', type=Path)
    parser.add_argument('--calibration-stage', type=Path)
    parser.add_argument('--scope', choices=['pilot4', 'remaining36'])
    parser.add_argument('--round-id')
    parser.add_argument('--reset-index',type=int,choices=[5,15,25],default=5)
    parser.add_argument('--pilot-stage', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_stage(args.audit_stage) if args.audit_stage else prepare(args)))
