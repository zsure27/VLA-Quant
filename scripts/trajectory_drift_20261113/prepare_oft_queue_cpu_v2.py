"""Compile disabled continuation templates; no authorization or GPU launch."""
import argparse
import copy
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import qvla.oft_quant_catalog_v1  # Explicit names only, not compatibility registration.
from qvla.infra.common import canonical, digest
from qvla.infra.config import validate
from qvla.infra.plan import compile_plan
from qvla.oft_prepared_queue_v1 import (
    build_command, require_current_authorization, validate_manifest,
)


def rejected(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('Unregistered CPU preparation was admitted')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--control', type=Path, required=True)
    a = p.parse_args()
    r, control = a.root.resolve(), a.control.resolve()
    code = Path(__file__).resolve().parents[2]
    out = control / 'prepared-queue-v2'
    out.mkdir(exist_ok=False)
    inventory = json.loads((control / 'inventory/result.json').read_text())
    assert inventory['gate'] == 'PASS_CPU_ASSETS_600_IDENTITIES_AND_240_PENDING_INITIAL25'
    cases = json.loads((control / 'inventory/pending-initial25.json').read_text())
    legacy = r / 'code-cpu-prep-20261109-v1'
    qstage = r / 'control/20261112-061-oft-qvla-resume-v6/stages/qvla-remaining9-v1'
    collectors = {
        'awq_w4a16': dict(path=str(legacy / 'qvla/execute_oft_quant_drift_v2.py')),
        'smoothquant_w4a8': dict(path=str(legacy / 'qvla/execute_oft_quant_drift_v2.py')),
        'smoothquant_w4a4': dict(path=str(legacy / 'qvla/execute_oft_quant_drift_v2.py')),
        'gptq_w4a16': dict(path=str(r / 'method-code/20261110-gptq-v3/execute_oft_gptq_v2.py')),
        'omniquant_w4a16': dict(path=str(r / 'method-code/20261110-next-stage-v1/execute_oft_omni_lwc_v1.py')),
        'qvla_w4a16': dict(path=str(r / 'code-20261112-061-qvla-resume-v5/scripts/trajectory_drift_20261112/collect_oft_qvla_case_v4.py'),
                           admission=str(qstage / 'admission/result.json')),
    }
    frozen = dict(inventory['source_frozen_files_verified'])
    for spec in collectors.values():
        frozen[spec['path']] = digest(Path(spec['path']))
        if spec.get('admission'):
            frozen[spec['admission']] = digest(Path(spec['admission']))
    # Preserve dependencies of independent GPTQ/Omni drivers, not just their entry point.
    for pth in [r / 'method-code/20261110-gptq-v3/gptq_oft_v1.py',
                r / 'method-code/20261110-omniquant-weight-v1/omni_oft_weight_v1.py',
                legacy / 'qvla/run_eval_official_quant.py', legacy / 'qvla/official_quant_adapter.py']:
        frozen[str(pth)] = digest(pth)
    for profile in inventory['profiles'].values():
        frozen[profile['path']] = profile['sha256']
    for case in cases:
        frozen[case['reference_result']] = case['reference_sha256']
    prepared_at = datetime.now(timezone.utc).isoformat()
    series = dict(format='OFT_EXISTING_METHODS_INITIAL25_CPU_PREP_V1', executable=False, root_control=None,
                  prepared_at=prepared_at, training='none', calibration_resets=[0],
                  checkpoint_receipt_sha256=inventory['checkpoint_receipt_sha256'],
                  profiles=inventory['profiles'], collectors=collectors, frozen_files=frozen,
                  cases=cases, actual_native_env_seed=0, model_seed=7, actions_per_query=8,
                  reason_not_executable='No current GPU authorization or UUID. Rebuild missing-case inventory '
                  'and compile independent round after user opens GPU. Never execute CPU template directly.')
    validate_manifest(series)
    bad = copy.deepcopy(series)
    bad['cases'].append(bad['cases'][0])
    rejected(lambda: validate_manifest(bad))
    config_fixture = dict(test='CPU contract only')
    auth = dict(gpu_execution_authorized=False)
    rejected(lambda: require_current_authorization(auth, series, Path('future-round'), config_fixture, 'host', 'gpu'))
    live = dict(series, executable=True)
    auth = dict(gpu_execution_authorized=True, round_id='future-round', config_sha256=canonical(config_fixture),
                hostname='host', gpu_uuid='gpu', granted_at=prepared_at)
    rejected(lambda: require_current_authorization(auth, live, Path('other-round'), config_fixture, 'host', 'gpu'))
    rejected(lambda: require_current_authorization(auth, live, Path('future-round'), config_fixture, 'host', 'other-gpu'))
    stale = dict(auth, granted_at=(datetime.fromisoformat(prepared_at) - timedelta(days=1)).isoformat())
    rejected(lambda: require_current_authorization(stale, live, Path('future-round'), config_fixture, 'host', 'gpu'))
    require_current_authorization(auth, live, Path('future-round'), config_fixture, 'host', 'gpu')
    for case in cases:
        cmd = build_command(r, series, case, Path('/future-round/case-runs') / case['condition'], 'cpu-template-only')
        assert str(Path(case['reference_result']).parent) in cmd if case['condition'] == 'qvla_w4a16' else '--initial-state-index' in cmd
    (out / 'series.json').write_text(json.dumps(series, indent=2) + '\n')
    base = json.loads((qstage / 'config.json').read_text())
    templates = []
    for condition in inventory['profiles']:
        stage = out / condition
        stage.mkdir()
        condition_series = dict(series, cases=[x for x in cases if x['condition'] == condition])
        (stage / 'series.json').write_text(json.dumps(condition_series, indent=2) + '\n')
        config = copy.deepcopy(base)
        method, activation = condition.split('_w')[0], int(condition.rsplit('a', 1)[1])
        config['quantization'].update(method=method, weight_bits=4, activation_bits=activation,
                                      profiles=[inventory['profiles'][condition]])
        config['evaluation']['suites'] = [dict(name=s, role='development', reset_offset=25, resets_per_task=1)
                                         for s in ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']]
        config['model']['oft_root'] = str(r / 'sources/openvla-oft')
        config['model']['libero_root'] = str(r / 'sources/LIBERO')
        config['finetuning'] = dict(method='none', train=False)
        config['binding'] = dict(id='oft-admitted-method-initial25-cpu-template-v1',
                                 verified_suites=[x['name'] for x in config['evaluation']['suites']],
                                 quantizers=[method], strategies=['none'],
                                 gate_files=[dict(path=str(control / 'inventory/result.json'),
                                                  sha256=digest(control / 'inventory/result.json'),
                                                  expected_gate=inventory['gate'])],
                                 evaluation_argv=[sys.executable, str(code / 'qvla/oft_prepared_queue_v1.py'),
                                                  '--root', str(r), '--round-dir', '{run_dir}',
                                                  '--suite', '{suite}', '--condition', condition],
                                 evaluation_audit_argv=[sys.executable,
                                      str(code / 'scripts/trajectory_drift_20261113/audit_prepared_pairs_v2.py'),
                                      '--root', str(r), '--series', '{run_dir}/series.json',
                                      '--round-dir', '{run_dir}', '--output', '{run_dir}/paired-audit'])
        config['scope'] = dict(executable=False, stage='CPU template only; no GPU authorization',
                               cases=40, same80_candidates_completed=True, no_old_runner_or_quota_override=True)
        config['control'].update(shutdown_provider='user_pause', defer_cloud_sync=True,
                                 heartbeat_seconds=900, stop_expansion_remaining_percent=15,
                                 close_remaining_percent=10,
                                 backup_argv=[sys.executable, str(code / 'scripts/infra_archive_run.py'),
                                              '--run-dir', '{run_dir}', '--prepare-only'],
                                 git_sync_argv=[sys.executable, '-c',
                                      "raise SystemExit('Consolidated sync belongs to the newly authorized controller')"])
        config['scope']['requires_fresh_root_shutdown_guard'] = True
        validate(config)
        (stage / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        # This is an actual infra compilation, not a runnable authorization.
        plan = compile_plan(config, code, stage)
        for path, sha in frozen.items():
            plan['inputs'].append(dict(path=path, sha256=sha, role='frozen scientific/dependency input'))
        plan['inputs'].append(dict(path=str(stage / 'series.json'), sha256=digest(stage / 'series.json')))
        plan.pop('plan_sha256')
        plan['plan_sha256'] = canonical(plan)
        (stage / 'CPU_COMPILED_NOT_AUTHORIZED_PLAN.json').write_text(json.dumps(plan, indent=2) + '\n')
        templates.append(dict(condition=condition, cases=40, stages=len(plan['stages']),
                              plan_sha256=plan['plan_sha256'], executable=False))
    result = dict(gate='PASS_CPU_SIX_INFRA_TEMPLATES_AND_AUTH_DEDUP_CONTRACTS',
                  total_pending=240, conditions=6, eval_suite_stages=24, templates=templates,
                  rejects=['CPU-only authorization', 'other round/config', 'other GPU',
                           'authorization predating preparation', 'duplicate cases'],
                  no_authorization_written=True, no_gpu_run_started=True,
                  full_runtime_verify_plan_deferred_to_fresh_gpu_registration=True)
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
