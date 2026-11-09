"""Audit measured native OFT traces; freeze BF16-repeat thresholds before candidates."""
from pathlib import Path
import json, hashlib
import numpy as np
from qvla.trajectory_diagnostics_v1 import rotation_difference, sustained_windows
from qvla.on_policy_capture import observation_hash
from experiments.robot.libero.run_libero_eval import process_action

ROOT = Path('runs')
OUT = Path('analysis/20261109-oft-pilot-v1')
SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
CONDITIONS = ['awq_w4a16', 'smoothquant_w4a8', 'smoothquant_w4a4']
REFS = {s: ROOT / ('20261108-053-oft-bf16-admission-v3' if s == 'libero_spatial' else f'20261108-053-oft-bf16-{s}-v1') for s in SUITES}
CACHE, AUDITS = {}, []
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    if str(p) in CACHE: return CACHE[str(p)]
    result = json.loads((p/'result.json').read_text())
    folder = p / ('episodes/task-%02d-reset-%03d' % (result['task_id'], result['initial_state_index']))
    m = json.loads((folder/'episode.json').read_text())
    rows = [json.loads(x) for x in (folder/'steps.jsonl').read_text().splitlines()]
    steps, queries = [[x for x in rows if x['kind'] == k] for k in ('control_step', 'policy_query')]
    assert len(steps) == m['control_steps'] == result['control_steps']
    assert len(queries) == m['policy_queries'] == result['policy_queries']
    assert m['status'] in ('success','timeout') and result['success'] == (m['status'] == 'success')
    chunks = {}
    for q, row in enumerate(queries):
        assert row['query_index'] == q and row['next_control_step'] == q*8
        with np.load(folder/f'query-{q:05d}.npz', allow_pickle=False) as z:
            assert observation_hash(z['image'], z['wrist_image'], z['state'], str(z['instruction'].item())) == row['input_sha256']
            a = z['raw_policy_chunk']; assert a.shape == (8,7) and np.isfinite(a).all(); chunks[q] = a.copy()
    for i, x in enumerate(steps):
        assert (x['control_step'], x['query_index'], x['chunk_position']) == (i,i//8,i%8)
        assert np.array_equal(x['controller_command'], process_action(chunks[i//8][i%8].copy(), 'openvla'))
        assert np.isfinite(x['post_position_m']).all() and np.isfinite(x['post_quaternion_xyzw']).all()
        if i: assert np.array_equal(x['pre_position_m'], steps[i-1]['post_position_m'])
        assert not x['done'] or (i == len(steps)-1 and m['status'] == 'success')
    assert bool(steps[-1]['done']) == (m['status'] == 'success')
    assert sha(p/'rollout.mp4') == result['video_sha256']
    if m['condition'] != 'bf16':
        profile = 'awq.pt' if m['condition'].startswith('awq') else 'smoothquant.pt'
        assert result['profile_sha256'] == sha(Path('profiles/20261109-four-suite-reset0-v4')/profile)
    value = dict(meta=m, result=result, steps=steps, first_input_sha=queries[0]['input_sha256'],
                 first_chunk=chunks[0], position=np.array([x['post_position_m'] for x in steps]), quat=np.array([x['post_quaternion_xyzw'] for x in steps]))
    CACHE[str(p)] = value
    AUDITS.append(dict(run=p.name, gate='PASS_QUERY_ACTION_POSE_VIDEO', steps=len(steps), queries=len(queries),
                       raw_env_seed_label=m['env_seed'], actual_native_env_seed=0,
                       note='Pinned native get_libero_env seeds 0; old BF16 label7 retained as metadata discrepancy'))
    return value
def compare(a,b):
    for k in ('suite','task_id','reset_index','model_seed','init_state_sha256','chunk_size','checkpoint_receipt_sha256','oft_source_commit'):
        assert a['meta'][k] == b['meta'][k], k
    assert a['first_input_sha'] == b['first_input_sha']
    n = min(len(a['steps']),len(b['steps']))
    v = a['position'][:n]-b['position'][:n]
    w = np.array([rotation_difference(x,y) for x,y in zip(a['quat'][:n], b['quat'][:n])])
    return np.linalg.norm(v,axis=1), np.linalg.norm(w,axis=1)*180/np.pi, w
