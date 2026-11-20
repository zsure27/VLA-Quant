"""Derived geometry from the audited four-suite pilot; no common-knee claim."""
import argparse
import json
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(path.read_text())


def poses(folder):
    positions, quats = [], []
    for file in sorted(folder.glob('step-*.npz')):
        with np.load(file, allow_pickle=False) as d:
            if not positions:
                positions.append(d['robot0_eef_pos'].copy())
                quats.append(d['robot0_eef_quat'].copy())
            positions.append(d['post_robot0_eef_pos'].copy())
            quats.append(d['post_robot0_eef_quat'].copy())
    p, q = np.asarray(positions), np.asarray(quats)
    assert len(p) > 1 and p.shape[1] == 3 and q.shape == (len(p), 4)
    assert np.isfinite(p).all() and np.isfinite(q).all()
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return p, q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    a.output.mkdir(exist_ok=False)
    suites = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
    traces, pairs, curves = {}, [], []
    for bits in (8, 4):
        stage = a.control / 'stages' / f'20261120-084-duquant-a{bits}-pilot4-v1'
        proof = load(stage / 'FULL_RAW_AUDIT.json')
        assert proof['gate'] == 'PASS_DUQUANT_REGISTERED_CASES_FULL_RAW_AUDIT' and proof['complete'] == 4
        for case in proof['rows']:
            folder = Path(case['result_path']).parent
            runtime = load(stage / 'case-plans' / (case['round_id'] + '-runtime.json'))
            reference = Path(runtime['bf16_reference'])
            result, ref = load(folder / 'result.json'), load(reference / 'result.json')
            assert result['suite'] == ref['suite'] == case['suite'] and result['task_id'] == ref['task_id'] == 0
            assert result['reset_index'] == ref['reset_index'] == 5
            p, q = poses(folder)
            bp, bq = poses(reference)
            assert np.array_equal(p[0], bp[0])
            n = min(len(p), len(bp))
            delta = (p[:n] - bp[:n]) * 1000
            translation = np.linalg.norm(delta, axis=1)
            rotation = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(q[:n] * bq[:n], axis=1)), 0, 1)))
            condition = f'duquant_w4a{bits}'
            pairs.append(dict(condition=condition, suite=case['suite'], task_id=0, reset_index=5,
                              success=bool(result['success']), bf16_success=bool(ref['success']),
                              control_steps=result['control_steps'], bf16_control_steps=ref['control_steps'],
                              common_post_steps=n-1, policy_queries=result['policy_queries'],
                              translation_p95_mm=float(np.percentile(translation[1:],95)),
                              rotation_p95_deg=float(np.percentile(rotation[1:],95)),
                              signed_translation_mean_mm=delta[1:].mean(axis=0).tolist(),
                              wall_seconds_including_load=result.get('wall_s_including_verification_and_load'),
                              wall_seconds_environment_and_queries=result.get('wall_s_env_and_queries')))
            traces['bf16',case['suite']] = bp
            traces[condition,case['suite']] = p
            curves.append(dict(condition=condition, suite=case['suite'], n_cases=1, control_step=list(range(n)),
                               translation_mm=translation.tolist(), rotation_deg=rotation.tolist()))
    summary = dict(gate='PASS_AUDITED_DUQUANT_PILOT_PAIRED_GEOMETRY', paired_cases=8,
                   configurations=2, tasks_per_suite=1, resets_per_task=1, pairs=pairs,
                   qualification='Admission examples only; not representative 10-task/5-initial coverage. Closed-loop separation includes feedback. No shared changepoint or OPD benefit established. Common-prefix errors stop at earlier termination, without padding.')
    (a.output / 'summary-public.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    (a.output / 'curves-public.json').write_text(json.dumps(curves,allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'bf16':'#222222','duquant_w4a8':'#db7219','duquant_w4a4':'#287caa'}
    fig = plt.figure(figsize=(17,13))
    views = [('3D',None),('XY',(0,1)),('XZ',(0,2)),('YZ',(1,2))]
    for row,(title,axes) in enumerate(views):
        for col,suite in enumerate(suites):
            ax = fig.add_subplot(4,4,row*4+col+1,projection='3d' if axes is None else None)
            for method in colors:
                xyz = traces[method,suite]
                if axes is None:
                    ax.plot(*xyz.T,color=colors[method],label=method,linewidth=1.4)
                    ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)')
                else:
                    i,j = axes
                    ax.plot(xyz[:,i],xyz[:,j],color=colors[method],label=method,linewidth=1.4)
                    ax.set(xlabel='xyz'[i]+' (m)',ylabel='xyz'[j]+' (m)')
                    ax.set_aspect('equal',adjustable='datalim')
                ax.grid(alpha=.2)
            ax.set_title(suite.replace('libero_','')+' / '+title,fontsize=10)
    handles,labels = ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,bbox_to_anchor=(.5,.972))
    fig.suptitle('Pi0.5 | task 0, reset 5 per suite | full paths to each own termination\nAdmission examples, not evidence of a common drift knee',fontsize=13,y=1.00)
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(a.output/'pilot-geometry-3d-2d.png',dpi=150,bbox_inches='tight');plt.close(fig)
    fig, axs = plt.subplots(2,4,figsize=(16,7),squeeze=False)
    for col,suite in enumerate(suites):
        for curve in [x for x in curves if x['suite']==suite]:
            for row,key in enumerate(['translation_mm','rotation_deg']):
                axs[row,col].plot(curve['control_step'],curve[key],color=colors[curve['condition']],label=curve['condition'])
                axs[row,col].grid(alpha=.2)
                axs[row,col].set_xlabel('Control step (common prefix only)')
        axs[0,col].set_title(suite+' | n=1 per method',fontsize=10)
    axs[0,0].set_ylabel('Position deviation from BF16 (mm)')
    axs[1,0].set_ylabel('SO(3) geodesic deviation (deg)')
    axs[0,0].legend(fontsize=8)
    fig.suptitle('Pi0.5 DuQuant pilot | no tail padding, no common-knee inference')
    fig.tight_layout();fig.savefig(a.output/'pilot-drift-by-control-step.png',dpi=150,bbox_inches='tight');plt.close(fig)
    print(json.dumps(dict(gate=summary['gate'],pairs=8,output=str(a.output))))


if __name__ == '__main__':
    main()
