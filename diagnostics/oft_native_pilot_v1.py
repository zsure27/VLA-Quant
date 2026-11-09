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
    folder = p/'episodes/task-00-reset-005'
    m, result = json.loads((folder/'episode.json').read_text()), json.loads((p/'result.json').read_text())
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
                 position=np.array([x['post_position_m'] for x in steps]), quat=np.array([x['post_quaternion_xyzw'] for x in steps]))
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
def main():
    OUT.mkdir(parents=True, exist_ok=False)
    repeats = [compare(load(ROOT/f'20261109-053-bf16-repeat-{s}-task0-reset5-v1'),load(REFS[s])) for s in SUITES]
    tp, rp = [float(np.percentile(np.concatenate([r[i] for r in repeats]),99)) for i in (0,1)]
    thresholds = dict(translation_m=max(.005,5*tp), rotation_deg=max(2.,5*rp), persistence=5,
                      repeat_translation_p99_m=tp, repeat_rotation_p99_deg=rp,
                      scope='Exploratory BF16-repeat thresholds with predeclared 5mm/2deg floors, not safety tolerances')
    (OUT/'frozen-thresholds.json').write_text(json.dumps(thresholds,indent=2))
    metrics, errors, data = [], {}, {}
    for s in SUITES:
        data[s,'bf16'] = load(REFS[s])
        for c in CONDITIONS:
            a = load(ROOT/f'20261109-053-{c}-{s}-task0-reset5-v1'); b = data[s,'bf16']
            t,r,w = compare(a,b); data[s,c] = a; errors[s,c] = (t,r,w)
            metrics.append(dict(suite=s,condition=c,task_id=0,reset_index=5,success=a['result']['success'],reference_success=b['result']['success'],
                candidate_steps=len(a['steps']),reference_steps=len(b['steps']),common_steps=len(t),translation_p95_mm=float(np.percentile(t,95)*1000),
                rotation_p95_deg=float(np.percentile(r,95)),translation_windows=sustained_windows(t,thresholds['translation_m'],5),
                rotation_windows=sustained_windows(r,thresholds['rotation_deg'],5),wall_s_including_load=a['result']['wall_s_including_load']))
    summary = dict(scope='Four tasks total; one task/reset per suite. No 40-task or five-state commonality claim.',thresholds=thresholds,metrics=metrics)
    (OUT/'raw-trace-audit.json').write_text(json.dumps(dict(gate='PASS_PAIRED_OFT_PILOT_RAW_AUDIT',episodes=AUDITS),indent=2))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    figures(data, errors, thresholds)
    print(json.dumps(summary))

def figures(data, errors, thresholds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#2878b5','#e07b39','#b44770']; files = []
    def save(fig,name):
        fig.suptitle('MEASURED OFT | task0/reset5 per suite | descriptive closed-loop differences',fontsize=11)
        fig.tight_layout(rect=(0,0,1,.96))
        for ext in ('png','pdf'): fig.savefig(OUT/f'{name}.{ext}',dpi=150)
        plt.close(fig); files.append(name+'.png')
    fig,axes=plt.subplots(2,4,figsize=(16,7))
    for j,s in enumerate(SUITES):
        for c,color in zip(CONDITIONS,colors):
            t,r,w=errors[s,c]
            axes[0,j].plot(t*1000,color=color,label=f'{c}, n={len(t)}'); axes[1,j].plot(r,color=color)
        axes[0,j].axhline(thresholds['translation_m']*1000,color='gray',ls='--'); axes[1,j].axhline(thresholds['rotation_deg'],color='gray',ls='--')
        axes[0,j].set_title(s);axes[0,j].legend(fontsize=6);axes[0,j].set_ylabel('Position difference (mm)');axes[1,j].set_ylabel('SO(3) difference (deg)');axes[1,j].set_xlabel('Control step; common observed domain')
    save(fig,'01-pose-difference-curves')
    fig=plt.figure(figsize=(15,11))
    for j,s in enumerate(SUITES):
        ax=fig.add_subplot(2,2,j+1,projection='3d')
        for c,color in [('bf16','black')]+list(zip(CONDITIONS,colors)):
            p=data[s,c]['position'];ax.plot(p[:,0],p[:,1],p[:,2],color=color,label=c,lw=1);ax.scatter(*p[-1],color=color,s=14)
        ax.set_title(s);ax.set_xlabel('World x (m)');ax.set_ylabel('World y (m)');ax.set_zlabel('World z (m)');ax.legend(fontsize=6)
    save(fig,'02-absolute-eef-trajectories')
    fig,axes=plt.subplots(4,3,figsize=(15,12))
    for j,s in enumerate(SUITES):
        for i,c in enumerate(CONDITIONS):
            w=errors[s,c][2]*180/np.pi
            for k,label in enumerate(('world x','world y','world z')):axes[j,i].plot(w[:,k],label=label,lw=.8)
            axes[j,i].axhline(0,color='gray',lw=.5);axes[j,i].set_title(s+' / '+c,fontsize=9);axes[j,i].set_ylabel('Signed SO(3) log (deg)');axes[j,i].set_xlabel('Control step')
    axes[0,0].legend(fontsize=7);save(fig,'03-signed-rotation-drift')
    fig,axes=plt.subplots(4,1,figsize=(13,10))
    for j,s in enumerate(SUITES):
        for c,color in [('bf16','black')]+list(zip(CONDITIONS,colors)):
            g=[x['controller_command'][-1] for x in data[s,c]['steps']];axes[j].step(np.arange(len(g)),g,where='post',color=color,label=c,alpha=.8)
        axes[j].set_title(s);axes[j].set_ylabel('Gripper command');axes[j].set_xlabel('Control step; command switch is not proof of physical grasp');axes[j].legend(fontsize=7,ncol=4)
    save(fig,'04-gripper-command-timing')
    (OUT/'index.html').write_text('<h1>Measured OFT four-suite pilot</h1><p>One task and one reset per suite. Matched-reset differences include feedback state divergence. No broad commonality or causal compensation conclusion.</p>'+''.join(f'<h2>{p}</h2><img style="width:100%" src="{p}">' for p in files))

if __name__ == '__main__': main()

