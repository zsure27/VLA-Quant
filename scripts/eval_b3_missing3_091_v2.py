"""Execute only task9 reset7-9 through the unchanged frozen M0 evaluator."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--frozen-source', type=Path, required=True)
    parser.add_argument('--resume-preflight', type=Path, required=True)
    parser.add_argument('--resume-preflight-sha256', required=True)
    parser.add_argument('--schedule-module', type=Path, required=True)
    extra, argv = parser.parse_known_args()
    sys.path.insert(0, str(extra.frozen_source))
    from qvla.b3_init_only_091_v1 import sha
    if sha(extra.resume_preflight) != extra.resume_preflight_sha256:
        raise ValueError('Recovery boundary receipt changed')
    gate = json.loads(extra.resume_preflight.read_text())
    if gate['gate'] != 'PASS_M0_47_PLUS_3_RECOVERY_PREFLIGHT':
        raise ValueError('No verified recovery partition')
    for name, expected in gate['frozen_files'].items():
        if sha(Path(name)) != expected:
            raise ValueError('Frozen material/source changed: ' + name)
    import importlib.util
    spec = importlib.util.spec_from_file_location('resume_schedule', extra.schedule_module)
    schedule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schedule)
    schedule.validate_partition([tuple(k) for k in gate['completed_pairs']],
                                [tuple(k) for k in gate['missing_pairs']])
    from qvla import run_eval_b3_init_only_091_v1 as runner
    args = runner.parse_args(argv)
    if args.candidate != 'svd-init-only' or (args.initial_state_offset, args.num_trials_per_task) != (7, 3):
        raise ValueError('Only missing S0 episodes are admitted')
    import experiments.robot.libero.run_libero_eval as L
    import numpy as np
    original_task, original_contract = L.run_task, runner.first_query_contract
    visited = []
    L.run_task = schedule.selected_run_task(original_task, visited)
    reference = gate['resume_first_query_reference']

    def bridge(model, cfg, observation, task, kwargs, action, record, state, initial):
        result = original_contract(model, cfg, observation, task, kwargs, action, record, state, initial)
        source = Path(reference['npz'])
        if sha(source) != reference['npz_sha256']:
            raise ValueError('PTQ matching first-query source changed')
        with np.load(source, allow_pickle=False) as sample:
            if (not np.array_equal(sample['image'], observation['full_image'])
                    or not np.array_equal(sample['wrist_image'], observation['wrist_image'])):
                raise ValueError('Resumed actual camera input differs from paired PTQ')
        query = reference['query']
        if (task != query['task'] or not np.array_equal(result['raw_state'], query['state'])
                or not np.array_equal(result['zero_branch_action'], query['raw_policy_chunk'])):
            raise ValueError('Resumed zero branch/raw input differs from matching PTQ action')
        receipt = {'gate': 'PASS_RESUMED_S0_ZERO_BRANCH_MATCHING_PTQ', 'task_id': 9,
                   'init_state_index': 7, 'paired_ptq_observation': str(source),
                   'paired_ptq_sha256': reference['npz_sha256'],
                   'actual_camera_equal': True, 'raw_proprio_equal': True,
                   'zero_action_bitwise_equal': True, 'preflight_sha256': extra.resume_preflight_sha256}
        (args.local_log_dir / 'recovery-first-query-bridge.json').write_text(json.dumps(receipt, indent=2) + '\n')
        return result

    runner.first_query_contract = bridge
    saved = sys.argv
    try:
        sys.argv = [str(extra.frozen_source / 'qvla/run_eval_b3_init_only_091_v1.py'), *argv]
        runner.main()
        if tuple(visited) != schedule.MISSING:
            raise ValueError('Incomplete scheduling execution')
        (args.local_log_dir / 'recovery-schedule-execution.json').write_text(json.dumps({
            'completed_selected_pairs': visited, 'old_complete_episodes_repeated': False,
            'preflight_sha256': extra.resume_preflight_sha256}, indent=2) + '\n')
    finally:
        sys.argv = saved
        L.run_task, runner.first_query_contract = original_task, original_contract


if __name__ == '__main__':
    main()
