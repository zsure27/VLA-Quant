"""Strict PTQ/S0 full 100-episode shard audit using unchanged scientific evaluator."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('control', 'frozen-source', 'root', 'output'):
        p.add_argument('--' + k, type=Path, required=True)
    p.add_argument('--offset', type=int, choices=(10, 20, 30, 40), required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.frozen_source))
    from audit_b3_segmented_091_v2 import segment
    from qvla.paired_metrics import comparison
    from qvla.b3_init_only_091_v1 import sha
    from scripts.audit_b3_init_only_091_v1 import validate_first_query_pair
    expected = [(t, r) for t in range(10) for r in range(a.offset, a.offset + 10)]
    records = {v: segment(a.root / v, expected, a.frozen_source) for v in ('ptq', 'svd-init-only')}
    ptq, s0 = records['ptq'], records['svd-init-only']
    if ptq['keys'] != s0['keys'] or ptq['first'] != s0['first']:
        raise ValueError('Task/reset/seed/initial input pairing differs')
    lock = a.control / ('M0-continuation-v2/reset-%d-%d/runtime-lock.json' % (a.offset, a.offset + 9))
    for v in records.values():
        if v['case']['invocation']['lock_sha256'] != sha(lock):
            raise ValueError('Actual invocation runtime lock differs')
        command = v['case']['invocation']['command']
        if command[1] != str(a.frozen_source / 'qvla/run_eval_b3_init_only_091_v1.py'):
            raise ValueError('Continuation did not use unchanged original scientific evaluator')
    validate_first_query_pair({v: r['case'] for v, r in records.items()})
    rows = [{'task_id': k[0], 'init_state_index': k[1],
             'ptq': ptq['success'][i], 'svd-init-only': s0['success'][i]} for i, k in enumerate(ptq['keys'])]
    costs = {}
    for v, record in records.items():
        wall = (a.root / v / 'exit-code.txt').stat().st_mtime - record['case']['invocation']['started_unix']
        if wall < 0:
            raise ValueError('Invalid timing evidence')
        costs[v] = {'queries': record['query_count'], 'wrapper_wall_seconds': wall,
                    'videos': record['videos'], 'optimization_steps': 0,
                    'wall_scope': 'Includes startup/checksum/admission/trace/video, not packed low-bit kernel speed'}
    result = {'gate': 'PASS_B3_INIT_ONLY_FULL_SHARD_PAIRED_PROTOCOL', 'offset': a.offset,
              'paired_episodes': 100, 'paired_rows': rows,
              'successes': {v: sum(r[v] for r in rows) for v in records},
              'comparison_S0_minus_PTQ': comparison(rows, 'svd-init-only', 'ptq'),
              'rescue_pairs': [r for r in rows if r['svd-init-only'] and not r['ptq']],
              'break_pairs': [r for r in rows if r['ptq'] and not r['svd-init-only']],
              'failures': {v: [r for r in rows if not r[v]] for v in records}, 'costs': costs,
              'classification': 'Same ten-task development; no micro overlap added and no independent-replication claim',
              'sources': {str(p): sha(p) for p in (a.frozen_source / 'qvla/run_eval_b3_init_only_091_v1.py',
                                                  a.frozen_source / 'qvla/b3_init_only_091_v1.py')},
              'next': 'Analyze cumulative rescue/break, task interval, McNemar, all failures and cost; fresh primary quota/disk then next registered shard'}
    if a.output.exists():
        raise ValueError('Immutable audit output exists')
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('gate', 'paired_episodes', 'successes', 'comparison_S0_minus_PTQ')}))


if __name__ == '__main__':
    main()
