"""Audit complete temporal-trained micro/first50 runs and their actual locked invocation."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

import os
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,os.environ['VLA_FROZEN_SOURCE'])


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def segment(directory, expected_pairs, source, allow_interrupted=False):
    import numpy as np
    from b1_temporal_eval_contract_091_v2 import sha, validate_provenance
    from run_eval_b1_temporal_091_v2 import parse_args
    from qvla.on_policy_capture import observation_hash
    from qvla.reproducibility import episode_seeds
    from scripts.audit_extended_peft_eval import normalized_trace_state, read_lines
    invocation = read(directory / 'invocation.json')
    command = invocation['command']
    own_root = Path(__file__).resolve().parent
    lock_path = own_root / 'first50-plan-v1/runtime-lock.json'
    if (invocation.get('lock_sha256') != sha(lock_path)
            or len(command) < 2 or command[1] != str(own_root / 'run_eval_b1_temporal_091_v2.py')):
        raise ValueError('Actual evaluator entry or preregistered runtime lock changed')
    args = parse_args(command[command.index('--candidate'):])
    if args.local_log_dir != directory:
        raise ValueError('Actual output differs from segment directory')
    resets = sorted({r for _, r in expected_pairs})
    if (args.initial_state_offset, args.num_trials_per_task) != (min(resets), len(resets)):
        raise ValueError('Actual reset offset/count differs from registered segment')
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
    candidate = read(directory / 'candidate-temporal.json')
    actual = read(directory / 'actual-model-temporal-gate.json')
    sources = {str(f): sha(f) for f in (Path(__file__).with_name('run_eval_b1_temporal_091_v2.py'), Path(__file__).with_name('b1_temporal_eval_contract_091_v2.py'))}
    if (candidate['sources'] != sources or candidate['checkpoint_identity'] != parent['checkpoint_identity']
            or candidate['trained_artifact_sha256'] != args.trained_artifact_sha256
            or candidate['adapter_state_sha256'] != sha(args.adapter_state)
            or actual['frozen_sha256'] != parent['frozen_before_sha256']):
        raise ValueError('Candidate does not bind the original scientific source/materials')
    return {'keys': keys, 'success': outcomes, 'first': first, 'references': references,
            'query_count': kept, 'unfinished_queries_excluded': len(eq) - kept,
            'videos': len(videos), 'frozen_files': frozen,
            'case': {'candidate': candidate, 'actual_gate': actual,
                     'first_query': queries[0], 'invocation': invocation}}



def main():
    p=argparse.ArgumentParser(description='Strict temporal Early80/Phase80 paired micro or first50')
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--count',type=int,choices=(1,5),required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import numpy as np
    from b1_temporal_eval_contract_091_v2 import MODELS,VERSION,sha
    from qvla.paired_metrics import comparison
    pairs=[(t,r) for t in range(10) for r in range(5,5+a.count)]
    records={v:segment(a.root/v,pairs,Path(os.environ['VLA_FROZEN_SOURCE'])) for v in MODELS}
    early,phase=records.values()
    if early['keys']!=phase['keys'] or early['first']!=phase['first']:
        raise ValueError('Temporal actual initial observation/seed pairing differs')
    for v,rec in records.items():
        cand,gate,q=rec['case']['candidate'],rec['case']['actual_gate'],rec['case']['first_query']
        if (cand['version']!=VERSION or cand['candidate']!=v or cand['model_id']!=MODELS[v]
                or cand['optimization_steps']!=1000 or cand['training_executed'] is not True
                or cand['parameters']!=19988480 or cand['rank']!=8 or cand['trained_state_used'] is not True):
            raise ValueError('Actual temporal trained role changed')
        if (gate['gate']!='PASS_B1_TEMPORAL_ACTUAL_MODEL' or gate['version']!=VERSION
                or gate['candidate']!=v or gate['optimization_steps']!=1000
                or not all(gate.get(k) is True for k in ('reload_output_equal','all_parameters_frozen',
                    'single_proprio_normalization_equal','matching_PTQ_input_and_zero_action_equal'))
                or gate['state_space']!='raw_proprio_before_get_action'
                or not np.array_equal(np.asarray(gate['raw_state']),np.asarray(q['state']))
                or not np.array_equal(np.asarray(gate['reference_action']),np.asarray(q['raw_policy_chunk']))):
            raise ValueError('Real first-query/normalization/reload/PTQ bridge incomplete')
        if not np.isfinite(np.asarray(gate['zero_branch_action'])).all():raise ValueError('Invalid zero action')
    for f in ('checkpoint_identity','profiles','sources'):
        if early['case']['candidate'][f]!=phase['case']['candidate'][f]:raise ValueError('Uncontrolled difference: '+f)
    eg,pg=early['case']['actual_gate'],phase['case']['actual_gate']
    if (eg['frozen_sha256']!=pg['frozen_sha256'] or eg['ptq_reference_sha256']!=pg['ptq_reference_sha256']
            or not np.array_equal(np.asarray(eg['zero_branch_action']),np.asarray(pg['zero_branch_action']))):
        raise ValueError('Temporal zero branches do not match same actual canonical PTQ')
    rows=[{'task_id':p[0],'init_state_index':p[1],**{v:records[v]['success'][i] for v in MODELS}} for i,p in enumerate(pairs)]
    costs={}
    for v,rec in records.items():
        wall=(a.root/v/'exit-code.txt').stat().st_mtime-rec['case']['invocation']['started_unix']
        if wall<0:raise ValueError('Invalid actual timing')
        costs[v]={'queries':rec['query_count'],'wrapper_wall_seconds':wall,'videos':rec['videos'],
                  'scope':'Includes loading/checksum/numeric gate/trace/video; not packed kernel speed'}
    result={'gate':'PASS_B1_TEMPORAL_PAIRED_PROTOCOL','paired_episodes':len(rows),'offset':5,'count':a.count,
            'paired_rows':rows,'successes':{v:sum(r[v] for r in rows) for v in MODELS},
            'comparison_Phase_minus_Early':comparison(rows,'phase80','early80'),
            'rescue_pairs':[r for r in rows if r['phase80'] and not r['early80']],
            'break_pairs':[r for r in rows if r['early80'] and not r['phase80']],
            'failures':{v:[r for r in rows if not r[v]] for v in MODELS},'costs':costs,
            'scope':'Same ten-task development; micro overlaps formal, do not add; training1000 fixed',
            'case_sources':{v:rec['case']['candidate'] for v,rec in records.items()}}
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ('gate','paired_episodes','successes','comparison_Phase_minus_Early')}))

if __name__=='__main__':main()
