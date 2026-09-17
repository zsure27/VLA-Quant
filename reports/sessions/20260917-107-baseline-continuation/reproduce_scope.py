"""Figures and provenance CSV from completed, manifest-paired scope experiments."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
RAW=HERE.parents[2]/'results/107-baseline-continuation-20260917'
summary=[];pairs=[]
for p in sorted(RAW.glob('awq-scope-shard-*/scope-paired-results.json')):
    d=json.loads(p.read_text());rs=d['paired_episodes'];case=d['case']
    s=dict(case=case,source_shard=p.parent.name,episodes=len(rs),successes=sum(r['candidate_success'] for r in rs),
        bf16_successes=sum(r['bf16_success'] for r in rs),episode_errors=sum(len(r['candidate_episode_errors']) for r in rs))
    summary.append(s)
    pairs.extend(dict(r,case=case,source_shard=p.parent.name) for r in rs)
    matrix=np.full((10,50),np.nan)
    for r in rs:matrix[r['task_id'],r['init_state_index']]=int(r['candidate_success'])-int(r['bf16_success'])
    covered=sorted(set(r['init_state_index'] for r in rs))
    fig,ax=plt.subplots(figsize=(10,4));ax.imshow(matrix[:,covered],vmin=-1,vmax=1,cmap='coolwarm',aspect='auto')
    ax.set_xticks(range(len(covered)));ax.set_xticklabels(covered);ax.set_yticks(range(10))
    ax.set_xlabel('Official initial state index');ax.set_ylabel('Spatial task ID')
    ax.set_title(case+': blue BF16 only; gray same; red candidate only')
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(HERE/'figures'/('scope-'+p.parent.name+'.'+ext),dpi=160)
    plt.close(fig)
if not summary:raise SystemExit('No completed scope pairs; no speculative plots')
keys=[(r['case'],r['task_id'],r['init_state_index']) for r in pairs]
assert len(keys)==len(set(keys)),'duplicate initial state within a candidate'
summary=[];per_task=[]
for case in sorted(set(r['case'] for r in pairs)):
    rs=[r for r in pairs if r['case']==case]
    summary.append(dict(case=case,source_shards=sorted(set(r['source_shard'] for r in rs)),
        episodes=len(rs),successes=sum(r['candidate_success'] for r in rs),
        bf16_successes=sum(r['bf16_success'] for r in rs),
        episode_errors=sum(len(r['candidate_episode_errors']) for r in rs)))
    for task in range(10):
        ts=[r for r in rs if r['task_id']==task]
        per_task.append(dict(case=case,task_id=task,episodes=len(ts),
            successes=sum(r['candidate_success'] for r in ts),bf16_successes=sum(r['bf16_success'] for r in ts)))
for name,rs in [('scope_summary',summary),('scope_paired_episodes',pairs),('scope_per_task',per_task)]:
    with (HERE/'data'/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
(HERE/'data/scope_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,ax=plt.subplots(figsize=(10,4));x=np.arange(len(summary))
ax.bar(x-.18,[r['bf16_successes']/r['episodes'] for r in summary],.36,label='Matched BF16')
ax.bar(x+.18,[r['successes']/r['episodes'] for r in summary],.36,label='Candidate')
labels={'language-w2-clip':'Language W2\nG128, clip','language-w2-no-clip':'Language W2\nG128, no clip',
    'language-w2-g64-clip':'Language W2\nG64, clip','language-w2-g64-no-clip':'Language W2\nG64, no clip',
    'vision-w2':'Vision W2\nD64 / S128'}
ax.set_xticks(x);ax.set_xticklabels([labels.get(r['case'],r['case'])+'\nn='+str(r['episodes']) for r in summary])
for i,r in enumerate(summary):ax.text(i+.18,r['successes']/r['episodes']+.02,str(r['successes'])+'/'+str(r['episodes']),ha='center',fontsize=9)
ax.set_ylim(0,1.05);ax.set_ylabel('Closed-loop success rate');ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2)
ax.set_title('Scope diagnostics; matched references; candidate coverage may differ',pad=40);fig.tight_layout()
for ext in ('png','svg'):fig.savefig(HERE/'figures'/('04_scope_success.'+ext),dpi=160)
plt.close(fig)
print(json.dumps(summary,indent=2))
