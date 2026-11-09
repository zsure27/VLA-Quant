"""Inventory completed native references and list missing cases; never dispatch."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-completed', type=int, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipt = hashlib.sha256((root / 'models/pi05_libero/DOWNLOAD_VERIFICATION.json').read_bytes()).hexdigest()
    completed, excluded = {}, []
    for file in sorted((root / 'runs').glob('*/result.json')):
        row = json.loads(file.read_text())
        if str(row.get('condition', '')).lower() != 'bf16':
            continue
        if row.get('gate') != 'PASS_PI05_NATIVE_BF16_EPISODE' or row.get('checkpoint_receipt_sha256') != receipt:
            excluded.append({'result': str(file.relative_to(root)), 'reason': 'not a passing exact-checkpoint native reference'})
            continue
        key = row['suite'], row['task_id'], row['reset_index']
        assert row['replan_steps'] == 5
        if key in completed:
            excluded.append({'result': str(file.relative_to(root)), 'reason': 'duplicate case'})
            continue
        completed[key] = {k: row[k] for k in ['suite', 'task_id', 'reset_index', 'gate', 'success',
                                             'control_steps', 'policy_queries', 'initial_state_sha256', 'replan_steps']}
        completed[key].update(result=str(file.relative_to(root)), result_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    assert len(completed) == args.expected_completed, len(completed)
    total_pending = 0
    for suite in ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']:
        pending = [{'task_id': task, 'reset_index': reset} for task in range(10)
                   for reset in [5, 15, 25, 35, 45] if (suite, task, reset) not in completed]
        total_pending += len(pending)
        config = {'model': 'pi05', 'condition': 'BF16', 'suite': suite, 'executable': False,
                  'gpu_run_admitted': False, 'native_execute_horizon': 5, 'checkpoint_receipt_sha256': receipt,
                  'pending_cases': pending, 'do_not_rerun_completed_cases': True,
                  'new_gpu_authorization_and_resource_gate_required': True}
        (out / (suite + '-bf16-resume.json')).write_text(json.dumps(config, indent=2))
    report = {'completed_unique': len(completed), 'planned_total': 200, 'pending': total_pending,
              'completed_cases': list(completed.values()), 'excluded': excluded,
              'scope': 'result metadata only; prior private raw-data audit remains authoritative; no new trajectory'}
    assert len(completed) + total_pending == 200
    (out / 'REFERENCE_RESUME_RECEIPT.json').write_text(json.dumps(report, indent=2))
    print('PASS_NATIVE_REFERENCE_RESUME_INVENTORY', len(completed), total_pending)


if __name__ == '__main__':
    main()
