"""Audit real completed episodes across an interrupted 47+3 execution; never forge exit0."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def segment(directory, expected_pairs, source, allow_interrupted=False):
    import numpy as np
    from qvla.b3_init_only_091_v1 import sha, validate_provenance
    from qvla.run_eval_b3_init_only_091_v1 import parse_args
    from qvla.on_policy_capture import observation_hash
    from qvla.reproducibility import episode_seeds
    from scripts.audit_extended_peft_eval import normalized_trace_state, read_lines
    invocation = read(directory / 'invocation.json')
    command = invocation['command']
    args = parse_args(command[command.index('--candidate'):])
    if args.local_log_dir != directory:
        raise ValueError('Actual output differs from segment directory')
    parent = validate_provenance(args)
    if allow_interrupted:
        if (directory / 'exit-code.txt').exists():
            raise ValueError('Original interrupted stage unexpectedly gained an exit receipt')
    elif (directory / 'exit-code.txt').read_text().strip() != '0':
        raise ValueError('Complete segment must have actual wrapper exit0')
    if invocation.get('hostname') != 'autodl-container-dfbc4bb0ff-1e85dd94':
        raise ValueError('Wrong actual execution host')
    logs = list(directory.glob('EVAL-*.txt'))
    if len(logs) != 1:
        raise ValueError('Ambiguous evaluator log')
    log = logs[0].read_text()
    console = (directory / 'console.log').read_text(errors='replace')
    if re.search(r'Episode error:|Traceback \(most recent call last\)', log + console):
        raise ValueError('Execution contains an exception')
    manifests = [json.loads(m) for m in re.findall(r'EPISODE_MANIFEST (\{.*\})', log)]
    outcomes = [m == 'True' for m in re.findall(r'Success: (True|False)', log)]
    events = read_lines(directory / 'on-policy-events.jsonl')
    queries = read_lines(directory / 'policy-queries.jsonl')
    starts = [e for e in events if e['record_type'] == 'episode_start']
    ends = [e for e in events if e['record_type'] == 'episode_end']
    eq = [e for e in events if e['record_type'] == 'query']
    n = len(expected_pairs)
    tail = 1 if allow_interrupted else 0
    if (len(starts) != n + tail or len(manifests) != n + tail
            or len(ends) != n or len(outcomes) != n
            or [e['episode_serial'] for e in starts] != list(range(n + tail))
            or [e['episode_serial'] for e in ends] != list(range(n))
            or any(e['aborted'] for e in ends)
            or outcomes != [e['success'] for e in ends]):
        raise ValueError('Completed-prefix boundary or event/log outcomes differ')
    if allow_interrupted and (manifests[-1]['task_id'], manifests[-1]['init_state_index']) != (9, 7):
        raise ValueError('Unexpected interrupted task/reset')
    keys = []
    for m, pair in zip(manifests, expected_pairs):
        if (m['task_id'], m['init_state_index']) != tuple(pair):
            raise ValueError('Actual coverage/order differs from registered task/reset')
        seed, env = episode_seeds(0, 1, 'libero_spatial', *pair)
        if (m['model_seed'], m['env_seed'], m['protocol']) != (seed, env, 'paired'):
            raise ValueError('Actual per-episode seeds changed')
        keys.append([*pair, m['init_state_sha256'], env, seed, 'paired'])
    if len(eq) != len(queries) or not eq:
        raise ValueError('Action/observation trace mismatch')
    norm = read(Path(args.pretrained_checkpoint) / 'dataset_statistics.json')['libero_spatial_no_noops']['proprio']
    next_query = {i: 0 for i in range(n + tail)}
    first, references, kept = {}, {}, 0
    frozen = {str(f): sha(f) for f in directory.rglob('*') if f.is_file()}
    for event, query in zip(eq, queries):
        serial = event['episode_serial']
        if (serial not in next_query or event['query_in_episode'] != next_query[serial]
                or event['chunk_execution_steps'] != 8
                or event['executed_step_start'] != next_query[serial] * 8):
            raise ValueError('Non-contiguous query or changed chunk execution')
        next_query[serial] += 1
        if serial >= n:
            continue  # Original unfinished episode is retained but never counted.
        action = np.asarray(query['raw_policy_chunk'])
        if query['finite'] is not True or action.shape != (8, 7) or not np.isfinite(action).all():
            raise ValueError('Malformed/nonfinite action')
        f = directory / event['file']
        if not f.resolve().is_relative_to(directory.resolve()) or sha(f) != event['file_sha256']:
            raise ValueError('Captured NPZ missing or changed')
        with np.load(f, allow_pickle=False) as z:
            fingerprint = observation_hash(z['image'], z['wrist_image'], z['state'], str(z['instruction'].item()))
            if (not np.array_equal(z['student_action'], action.astype(np.float32))
                    or query.get('state_space') != 'raw_proprio_before_get_action'
                    or not np.array_equal(z['state'], normalized_trace_state(query['state'], norm))
                    or event['task'] != query['task'] or query['task'] != str(z['instruction'].item())):
                raise ValueError('Captured action/task/normalization differs from execution')
            for k in z.files:
                if z[k].dtype.kind in 'fci' and not np.isfinite(z[k]).all():
                    raise ValueError('Nonfinite saved numeric content')
        if fingerprint != event['observation_sha256']:
            raise ValueError('Observation content fingerprint changed')
        if event['query_in_episode'] == 0:
            first[serial] = fingerprint
            references[serial] = {'query': query, 'npz': str(f), 'npz_sha256': event['file_sha256']}
        kept += 1
    if (set(first) != set(range(n))
            or any(e['queries'] != next_query[e['episode_serial']] for e in ends)):
        raise ValueError('Missing first query/completed query counts')
    stamp = re.search(r'\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}', logs[0].name).group()
    videos = sorted((source / 'rollouts').rglob(stamp + '--*.mp4'))
    seen = set()
    for f in videos:
        m = re.search(r'--episode=(\d+)--success=(True|False)--', f.name)
        if m is None or not 1 <= int(m[1]) <= n or int(m[1]) in seen or f.stat().st_size == 0:
            raise ValueError('Video ordinal/coverage invalid')
        index = int(m[1]) - 1
        if (m[2] == 'True') != outcomes[index]:
            raise ValueError('Video outcome differs from event/log')
        seen.add(index + 1)
        frozen[str(f)] = sha(f)
    if seen != set(range(1, n + 1)):
        raise ValueError('Original completed video coverage incomplete')
    candidate = read(directory / 'candidate-init-only.json')
    actual = read(directory / 'actual-model-init-only-gate.json')
    sources = {name: sha(source / name) for name in ('qvla/run_eval_b3_init_only_091_v1.py', 'qvla/b3_init_only_091_v1.py')}
    if (candidate['sources'] != sources or candidate['checkpoint_identity'] != parent['checkpoint_identity']
            or candidate['parent_artifact_sha256'] != args.initial_artifact_sha256
            or candidate['initial_state_sha256'] != sha(args.initial_state)
            or actual['frozen_sha256'] != parent['frozen_before_sha256']):
        raise ValueError('Candidate does not bind the original scientific source/materials')
    return {'keys': keys, 'success': outcomes, 'first': first, 'references': references,
            'query_count': kept, 'unfinished_queries_excluded': len(eq) - kept,
            'videos': len(videos), 'frozen_files': frozen,
            'case': {'candidate': candidate, 'actual_gate': actual,
                     'first_query': queries[0], 'invocation': invocation}}


def preflight(old_control, source, old_session):
    from scripts.audit_b3_init_only_091_v1 import validate_first_query_pair
    from qvla.b3_init_only_091_v1 import sha
    expected = [(t, r) for t in range(10) for r in range(5, 10)]
    ptq = segment(old_session / 'first50/ptq', expected, source)
    s0 = segment(old_session / 'first50/svd-init-only', expected[:47], source, True)
    if ptq['keys'][:47] != s0['keys'] or list(ptq['first'].values())[:47] != list(s0['first'].values()):
        raise ValueError('Completed 47 do not pair exactly with original PTQ')
    validate_first_query_pair({'ptq': ptq['case'], 'svd-init-only': s0['case']})
    old_lock = old_control / 'M0-plans-v1/runtime-lock.json'
    for case in (ptq, s0):
        if case['case']['invocation']['lock_sha256'] != sha(old_lock):
            raise ValueError('Original invocation lock mismatch')
    frozen = read(old_lock)['files']
    for f, h in frozen.items():
        if sha(Path(f)) != h:
            raise ValueError('Original lock material/source changed: ' + f)
    frozen.update(ptq['frozen_files']); frozen.update(s0['frozen_files'])
    return {'gate': 'PASS_M0_47_PLUS_3_RECOVERY_PREFLIGHT', 'completed_pairs': expected[:47],
            'missing_pairs': expected[47:], 'ptq_complete': 50, 's0_complete': 47,
            'old_S0_wrapper_exit0': False, 'unfinished_original_queries_excluded': s0['unfinished_queries_excluded'],
            'resume_first_query_reference': ptq['references'][47],
            'frozen_files': frozen, 'old_lock_sha256': sha(old_lock),
            'scientific_scope': 'Original 47 completed episodes verified; incomplete task9/reset7 suffix excluded, retained unchanged'}


def combined(old_control, source, old_session, resumed, receipt):
    import numpy as np
    from qvla.b3_init_only_091_v1 import sha
    from qvla.paired_metrics import comparison
    from scripts.audit_b3_init_only_091_v1 import validate_first_query_pair
    from qvla.b3_init_only_091_v1 import MODELS
    for f, h in receipt['frozen_files'].items():
        if sha(Path(f)) != h:
            raise ValueError('Original completed data/material changed after preflight')
    expected = [(t, r) for t in range(10) for r in range(5, 10)]
    ptq = segment(old_session / 'first50/ptq', expected, source)
    old = segment(old_session / 'first50/svd-init-only', expected[:47], source, True)
    new = segment(resumed, expected[47:], source)
    bridge = read(resumed / 'recovery-first-query-bridge.json')
    execution = read(resumed / 'recovery-schedule-execution.json')
    if (bridge['gate'] != 'PASS_RESUMED_S0_ZERO_BRANCH_MATCHING_PTQ'
            or execution['completed_selected_pairs'] != [[9, 7], [9, 8], [9, 9]]
            or execution['old_complete_episodes_repeated'] is not False):
        raise ValueError('Missing actual recovery scheduling/numerical bridge gate')
    validate_first_query_pair({'ptq': ptq['case'], 'svd-init-only': old['case']})
    synthetic_reference = dict(ptq['case'], first_query=ptq['references'][47]['query'])
    # Same normalization/reload/action shape checks on new actual first query; compare
    # zero branch to PTQ at its matching state, rather than global episode zero.
    actual = new['case']['actual_gate']
    if (not all(actual.get(k) is True for k in ('reload_output_equal', 'all_parameters_frozen', 'single_proprio_normalization_equal'))
            or actual['optimization_steps'] != 0
            or not np.array_equal(actual['reference_action'], new['case']['first_query']['raw_policy_chunk'])
            or not np.array_equal(actual['zero_branch_action'], synthetic_reference['first_query']['raw_policy_chunk'])):
        raise ValueError('Recovered first query fails role/numerical pairing')
    if new['case']['candidate'] != old['case']['candidate']:
        raise ValueError('Recovered candidate is not the unchanged S0 role')
    s_keys = old['keys'] + new['keys']
    s_first = list(old['first'].values()) + list(new['first'].values())
    if ptq['keys'] != s_keys or list(ptq['first'].values()) != s_first:
        raise ValueError('Segmented 50 task/reset/seeds/input content not strictly paired')
    labels = old['success'] + new['success']
    rows = [{'task_id': k[0], 'init_state_index': k[1], 'ptq': p, 'svd-init-only': s}
            for k, p, s in zip(ptq['keys'], ptq['success'], labels)]
    stop = read(old_control / 'quota-stop-receipt.json')
    old_wall = datetime.fromisoformat(stop['time_utc']).timestamp() - old['case']['invocation']['started_unix']
    new_wall = (resumed / 'exit-code.txt').stat().st_mtime - new['case']['invocation']['started_unix']
    return {'gate': 'PASS_B3_INIT_ONLY_SEGMENTED_PAIRED_PROTOCOL', 'phase': 'first50-recovered',
            'paired_episodes': 50, 'paired_rows': rows,
            'successes': {v: sum(row[v] for row in rows) for v in MODELS},
            'comparison_S0_minus_PTQ': comparison(rows, 'svd-init-only', 'ptq'),
            'rescue_pairs': [r for r in rows if r['svd-init-only'] and not r['ptq']],
            'break_pairs': [r for r in rows if r['ptq'] and not r['svd-init-only']],
            'failures': {v: [r for r in rows if not r[v]] for v in MODELS},
            'costs': {'ptq_queries': ptq['query_count'], 's0_completed_queries': old['query_count'] + new['query_count'],
                      's0_interrupted_queries_excluded': old['unfinished_queries_excluded'],
                      's0_old_attempt_wall_seconds_including_interruption': old_wall,
                      's0_recovery_wall_seconds': new_wall, 'videos_verified': 100,
                      'scope': 'Includes startup/checksum/admission/trace/video and wasted partial episode; no packed-INT2 speed claim'},
            'old_s0_wrapper_exit0': False, 'recovery_wrapper_exit0': True,
            'runtime_locks': {'original': ptq['case']['invocation']['lock_sha256'],
                              'recovery': new['case']['invocation']['lock_sha256']},
            'sources_and_candidate_equal_across_segments': True,
            'classification': 'Same ten-task development; exact episode partition, no extra sample or independent replication claim',
            'next': 'Analyze all paired failures/costs, fresh primary quota/disk; registered reset10-19 next without score-direction selection'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('frozen-source', 'old-control', 'old-session', 'output'):
        p.add_argument('--' + k, type=Path, required=True)
    p.add_argument('--resumed', type=Path)
    p.add_argument('--preflight', type=Path)
    a = p.parse_args()
    sys.path.insert(0, str(a.frozen_source))
    if a.output.exists():
        raise ValueError('Immutable audit output exists')
    result = (combined(a.old_control, a.frozen_source, a.old_session, a.resumed, read(a.preflight))
              if a.resumed else preflight(a.old_control, a.frozen_source, a.old_session))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('gate', 'paired_episodes', 'successes', 'comparison_S0_minus_PTQ') if k in result}))


if __name__ == '__main__':
    main()
