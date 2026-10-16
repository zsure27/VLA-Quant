"""Create equal-size Early80/Phase80 from the same complete, frozen A4 rollouts."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_queries(count, variant):
    if count < 2:
        raise ValueError('Two observations require a complete trajectory with >=2 queries')
    if variant == 'early80':
        return (0, 1)
    if variant != 'phase80':
        raise ValueError('Unknown temporal cohort')
    result = tuple(int(round(p * (count - 1))) for p in (.25, .75))
    if result[0] == result[1]:
        raise ValueError('Temporal positions collapsed')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'parent-training', 'frozen-source', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.frozen_source))
    import numpy as np
    from qvla.on_policy_capture import observation_hash
    from qvla.extended_peft import canonical_sha
    from qvla.data_roles import audit_training_inputs
    old = json.loads((a.parent_training / 'manifest.json').read_text())
    invocation = json.loads((a.run / 'invocation.json').read_text())
    cmd = invocation['command']
    for flag, expected in (('--model-id', 'A4'), ('--initial-state-offset', '0'),
                           ('--num_trials_per_task', '4'), ('--seed-protocol', 'paired'),
                           ('--seed', '0'), ('--env-seed', '1')):
        if cmd[cmd.index(flag) + 1] != expected:
            raise ValueError('Original A4 training capture identity changed: ' + flag)
    if (a.run / 'exit-code.txt').read_text().strip() != '0':
        raise ValueError('Original capture did not finish')
    logs = list(a.run.glob('EVAL-*.txt'))
    if len(logs) != 1:
        raise ValueError('Ambiguous original episode manifest')
    pairs = [json.loads(m.group(1)) for m in re.finditer(r'EPISODE_MANIFEST (\{.*\})', logs[0].read_text())]
    expected = {(t, r) for t in range(10) for r in range(4)}
    if len(pairs) != 40 or {(r['task_id'], r['init_state_index']) for r in pairs} != expected:
        raise ValueError('Need the original40 training episodes once')
    provenance = {n: sha(a.run / n) for n in
                  ('invocation.json', 'on-policy-events.jsonl', 'policy-queries.jsonl', logs[0].name)}
    if canonical_sha(provenance) != old['source_run_sha256']:
        raise ValueError('Capture is not the parent Early80 source')
    events = [json.loads(l) for l in (a.run / 'on-policy-events.jsonl').read_text().splitlines()]
    ends = [e for e in events if e['record_type'] == 'episode_end']
    if len(ends) != 40 or {e['episode_serial'] for e in ends} != set(range(40)) or any(e['aborted'] for e in ends):
        raise ValueError('Incomplete/aborted trajectory; do not filter by success')
    groups = {i: [] for i in range(40)}
    source_hashes = {}
    for e in events:
        if e['record_type'] != 'query':
            continue
        f = a.run / e['file']
        if not f.resolve().is_relative_to(a.run.resolve()) or sha(f) != e['file_sha256']:
            raise ValueError('Source query SHA/path mismatch')
        source_hashes[e['file']] = e['file_sha256']
        groups[e['episode_serial']].append(e)
    for e in ends:
        q = groups[e['episode_serial']]
        if [x['query_in_episode'] for x in q] != list(range(e['queries'])):
            raise ValueError('Non-contiguous source queries')
    parent = {(r['episode_serial'], r['query_in_episode']): r for r in old['samples']}
    if len(parent) != 80:
        raise ValueError('Wrong parent training set')
    if a.output.exists():
        raise ValueError('Never overwrite temporal training data')
    a.output.mkdir(parents=True)
    cohorts = {}
    for variant in ('early80', 'phase80'):
        out = a.output / variant
        out.mkdir()
        rows = []
        for serial in range(40):
            trajectory = groups[serial]
            pair = pairs[serial]
            for slot, query in enumerate(select_queries(len(trajectory), variant)):
                event = trajectory[query]
                with np.load(a.run / event['file'], allow_pickle=False) as sample:
                    image, wrist = sample['image'], sample['wrist_image']
                    state = sample['state'].astype(np.float32)
                    action = sample['student_action'].astype(np.float32)
                    instruction = str(sample['instruction'].item())
                if state.shape != (8,) or action.shape != (8, 7) or not np.isfinite(state).all() or not np.isfinite(action).all():
                    raise ValueError('Invalid normalized state or real action chunk')
                for im in (image, wrist):
                    if im.dtype != np.uint8 or im.ndim != 3 or im.shape[-1] != 3:
                        raise ValueError('Invalid camera input')
                if observation_hash(image, wrist, state, instruction) != event['observation_sha256']:
                    raise ValueError('Observation content mismatch')
                if variant == 'early80':
                    prior = parent[(serial, query)]
                    if prior['source_sha256'] != event['file_sha256'] or prior['observation_sha256'] != event['observation_sha256']:
                        raise ValueError('Early input no longer equals original training input')
                    oldfile = a.parent_training / prior['file']
                    if sha(oldfile) != prior['sample_sha256']:
                        raise ValueError('Parent sample SHA mismatch')
                    with np.load(oldfile, allow_pickle=False) as z:
                        if any(not np.array_equal(z[k], val) for k, val in
                               (('image', image), ('wrist_image', wrist), ('state', state), ('instruction', np.array(instruction)))):
                            raise ValueError('Early80 content differs from parent')
                target = out / ('sample-%05d.npz' % len(rows))
                np.savez_compressed(target, image=image, wrist_image=wrist, state=state,
                                    instruction=np.array(instruction), action=action[0])
                rows.append({'file': target.name, 'sample_sha256': sha(target),
                             'task_id': pair['task_id'], 'init_state_index': pair['init_state_index'],
                             'init_state_sha256': pair['init_state_sha256'], 'episode_serial': serial,
                             'query_in_episode': query, 'slot_in_episode': slot,
                             'trajectory_queries': len(trajectory), 'source': event['file'],
                             'source_sha256': event['file_sha256'], 'observation_sha256': event['observation_sha256']})
        manifest = {'schema_version': 'b1-temporal80-v1', 'variant': variant,
                    'role': 'student_state_train', 'state_space': 'policy_normalized_proprio',
                    'source_model_id': 'A4', 'count': 80, 'samples': rows,
                    'source_run': str(a.run), 'source_run_sha256': old['source_run_sha256'],
                    'parent_training': str(a.parent_training), 'parent_manifest_sha256': sha(a.parent_training / 'manifest.json'),
                    'selection': 'query0/1' if variant == 'early80' else 'round(p*(queries-1)), p=.25,.75',
                    'success_filtering': False, 'holdout_touched': False,
                    'dataset_action_role': 'placeholder first action; fresh same-observation BF16 teacher required'}
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        audit_training_inputs(sorted(out.glob('sample-*.npz')), None)
        cohorts[variant] = {'manifest': str(out / 'manifest.json'), 'sha256': sha(out / 'manifest.json')}
    receipt = {'gate': 'PASS_B1_TEMPORAL80_SOURCE_AND_CONTENT', 'cohorts': cohorts,
               'source_queries_verified': len(source_hashes), 'source_query_sha256': source_hashes,
               'source_provenance': provenance, 'episodes': 40, 'per_cohort': 80,
               'early80_parent_content_equal': True, 'success_filtering': False,
               'holdout_touched': False, 'gpu_used': False}
    (a.output / 'data-gate.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'source_query_sha256'}))


if __name__ == '__main__':
    main()
