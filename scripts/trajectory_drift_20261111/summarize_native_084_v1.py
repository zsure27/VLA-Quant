"""Summarize completed native references; never call baseline motion quantization drift."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--round-dir',type=Path,required=True)
a=p.parse_args()
out=a.round_dir/'native-reference-summary'
out.mkdir(exist_ok=False)
clone=json.loads((a.round_dir/'clone-verification.json').read_text())
roots=[Path(x['result']).parent for x in clone['completed']]
roots+=sorted((a.round_dir/'case-runs').iterdir())
seen={};traces={};rows=[]
for root in roots:
    x=json.loads((root/'result.json').read_text())
    assert x['gate']=='PASS_PI05_NATIVE_BF16_EPISODE' and x['reset_index']==5
    key=x['suite'],x['task_id'];assert key not in seen
    seen[key]=root
    steps=sorted(root.glob('step-*.npz'));assert len(steps)==x['control_steps']
    poses=[]
    for f in steps:
        with np.load(f,allow_pickle=False) as n:poses.append(n['robot0_eef_pos'].copy())
    with np.load(steps[-1],allow_pickle=False) as n:poses.append(n['post_robot0_eef_pos'].copy())
    poses=np.asarray(poses);assert poses.shape==(len(steps)+1,3) and np.isfinite(poses).all()
    traces[key]=poses
    rows.append(dict(suite=key[0],task_id=key[1],reset_index=5,success=x['success'],
                     control_steps=x['control_steps'],policy_queries=x['policy_queries'],
                     observed_eef_path_mm=float(np.linalg.norm(np.diff(poses,axis=0),axis=1).sum()*1000),
                     observed_start_to_end_mm=float(np.linalg.norm(poses[-1]-poses[0])*1000),
                     result=str(root/'result.json'),new_in_this_round=root.is_relative_to(a.round_dir)))
suites=['libero_spatial','libero_object','libero_goal','libero_10']
assert set(seen)=={(s,t) for s in suites for t in range(10)}
summary=dict(gate='PASS_PI05_INITIAL5_40_NATIVE_REFERENCE_SUMMARY',complete_unique=40,
             successes=sum(x['success'] for x in rows),new_complete=sum(x['new_in_this_round'] for x in rows),
             new_successes=sum(x['success'] for x in rows if x['new_in_this_round']),
             quantized_episodes=0,quantization_drift_conclusion_available=False,
             suites={s:dict(complete=10,success=sum(x['success'] for x in rows if x['suite']==s)) for s in suites},
             cases=rows,note='Task diversity at a single initial state index; five-initial study not completed. Native BF16 motion is not quantization drift.')
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
colors=plt.get_cmap('tab10')
fig=plt.figure(figsize=(15,10))
for i,suite in enumerate(suites):
    ax=fig.add_subplot(2,2,i+1,projection='3d')
    for task in range(10):
        pos=traces[(suite,task)]*1000
        ax.plot(*pos.T,color=colors(task),linewidth=1.25,label='task '+str(task))
        ax.scatter(*pos[0],color=colors(task),s=10,marker='o')
    ax.set_title(suite);ax.set_xlabel('X / mm');ax.set_ylabel('Y / mm');ax.set_zlabel('Z / mm')
    ax.view_init(elev=25,azim=-60)
fig.suptitle('MEASURED pi0.5 BF16 reference paths | 4 suites x 10 tasks | initial 5\nNative baseline coverage only: no quantization comparison',fontsize=14)
fig.legend(*fig.axes[0].get_legend_handles_labels(),loc='lower center',ncol=10,fontsize=8)
fig.tight_layout(rect=(0,0.05,1,.94));fig.savefig(out/'01-native-reference-3d.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(14,5))
for i,suite in enumerate(suites):
    data=sorted([x for x in rows if x['suite']==suite],key=lambda x:x['task_id'])
    axs[0].plot(range(10),[x['control_steps'] for x in data],marker='o',label=suite)
    axs[1].plot(range(10),[x['observed_eef_path_mm'] for x in data],marker='o',label=suite)
for ax in axs:ax.set_xlabel('Task index');ax.set_xticks(range(10));ax.grid(alpha=.25)
axs[0].set_ylabel('Executed control steps');axs[1].set_ylabel('Observed EEF path length / mm');axs[0].legend(fontsize=8)
fig.suptitle('MEASURED pi0.5 BF16 reference workload | initial 5 only | no quantization drift estimate')
fig.tight_layout();fig.savefig(out/'02-native-reference-workload.png',dpi=160);plt.close(fig)
print(json.dumps({k:v for k,v in summary.items() if k!='cases'}))
