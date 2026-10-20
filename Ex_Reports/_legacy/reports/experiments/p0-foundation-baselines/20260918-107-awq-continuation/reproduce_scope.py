"""Figures and provenance CSV from completed, manifest-paired scope experiments."""
import csv,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
RAWS=[HERE.parents[2]/'results/experiments/p0-foundation-baselines/20260917-107-baseline-continuation',HERE.parents[2]/'results/experiments/p0-foundation-baselines/20260918-107-awq-continuation']
summary=[];pairs=[];contracts={};evaluator_contract=None
def paired_stats(rs):
    a=sum(r['bf16_success'] and not r['candidate_success'] for r in rs)
    b=sum(r['candidate_success'] and not r['bf16_success'] for r in rs);n=a+b
    p=min(1.,2.*sum(math.factorial(n)//(math.factorial(k)*math.factorial(n-k)) for k in range(min(a,b)+1))/2**n) if n else 1.
    return dict(bf16_only=a,candidate_only=b,discordant_pairs=n,mcnemar_exact_p=p,
        both_success=sum(r['bf16_success'] and r['candidate_success'] for r in rs),
        both_fail=sum(not r['bf16_success'] and not r['candidate_success'] for r in rs))
for p in sorted(p for root in RAWS for p in root.glob('awq-scope-shard-*/scope-paired-results.json')):
    d=json.loads(p.read_text());rs=d['paired_episodes'];case=d['case']
    contract=(p.parent/'CONTRACT_SHA256SUMS.txt').read_text().splitlines()
    if evaluator_contract is None:evaluator_contract=contract[:2]
    assert contract[:2]==evaluator_contract,'evaluator contract changed'
    assert contracts.setdefault(case,contract)==contract,'profile contract changed within candidate'
    assert all(0<=r['task_id']<10 and 0<=r['init_state_index']<50 for r in rs)
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
    for ext in ('png','svg'):
        filename='scope-'+p.parent.name+'.'+ext
        original=HERE.parent/'20260917-107-baseline-continuation'/'figures'/filename
        if p.parent.parent==RAWS[0] and original.is_file():continue
        fig.savefig(HERE/'figures'/filename,dpi=160)
    plt.close(fig)
if not summary:raise SystemExit('No completed scope pairs; no speculative plots')
keys=[(r['case'],r['task_id'],r['init_state_index']) for r in pairs]
assert len(keys)==len(set(keys)),'duplicate initial state within a candidate'
(HERE/'data/scope_contracts.json').write_text(json.dumps(contracts,indent=2)+'\n')
summary=[];per_task=[]
for case in sorted(set(r['case'] for r in pairs)):
    rs=[r for r in pairs if r['case']==case]
    summary.append(dict(case=case,source_shards=sorted(set(r['source_shard'] for r in rs)),
        episodes=len(rs),successes=sum(r['candidate_success'] for r in rs),
        bf16_successes=sum(r['bf16_success'] for r in rs),
        episode_errors=sum(len(r['candidate_episode_errors']) for r in rs),**paired_stats(rs)))
    for task in range(10):
        ts=[r for r in rs if r['task_id']==task]
        per_task.append(dict(case=case,task_id=task,episodes=len(ts),
            successes=sum(r['candidate_success'] for r in ts),bf16_successes=sum(r['bf16_success'] for r in ts)))
for name,rs in [('scope_summary',summary),('scope_paired_episodes',pairs),('scope_per_task',per_task)]:
    with (HERE/'data'/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
(HERE/'data/scope_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,ax=plt.subplots(figsize=(max(10,2.3*len(summary)),4.5));x=np.arange(len(summary))
ax.bar(x-.18,[r['bf16_successes']/r['episodes'] for r in summary],.36,label='Matched BF16')
ax.bar(x+.18,[r['successes']/r['episodes'] for r in summary],.36,label='Candidate')
labels={'language-w2-clip':'Language W2\nG128, clip','language-w2-no-clip':'Language W2\nG128, no clip',
    'language-w2-g64-clip':'Language W2\nG64, clip','language-w2-g64-no-clip':'Language W2\nG64, no clip',
    'vision-w2':'Vision W2\nD64 / S128',
    'language-w2-g64-no-clip-attention':'Language W2\nG64, attention no clip',
    'language-w2-g64-no-clip-mlp':'Language W2\nG64, MLP no clip'}
ax.set_xticks(x);ax.set_xticklabels([labels.get(r['case'],r['case'])+'\nn='+str(r['episodes']) for r in summary])
for i,r in enumerate(summary):ax.text(i+.18,r['successes']/r['episodes']+.02,str(r['successes'])+'/'+str(r['episodes']),ha='center',fontsize=9)
ax.set_ylim(0,1.05);ax.set_ylabel('Closed-loop success rate');ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2)
ax.set_title('Scope diagnostics; matched references; candidate coverage may differ',pad=40);fig.tight_layout()
for ext in ('png','svg'):fig.savefig(HERE/'figures'/('04_scope_success.'+ext),dpi=160)
plt.close(fig)
vision=[r for r in per_task if r['case']=='vision-w2']
if vision:
    fig,ax=plt.subplots(figsize=(10,4));x=np.arange(len(vision))
    ax.bar(x-.18,[r['bf16_successes']/r['episodes'] for r in vision],.36,label='Matched BF16')
    ax.bar(x+.18,[r['successes']/r['episodes'] for r in vision],.36,label='Vision W2')
    for i,r in enumerate(vision):ax.text(i+.18,r['successes']/r['episodes']+.02,str(r['successes'])+'/'+str(r['episodes']),ha='center',fontsize=9)
    ax.set_xticks(x);ax.set_xticklabels([r['task_id'] for r in vision]);ax.set_ylim(0,1.13)
    ax.set_xlabel('Spatial task ID');ax.set_ylabel('Closed-loop success rate');ax.legend(loc='lower left')
    ax.set_title('Vision W2 task sensitivity; same official states and seeds');fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(HERE/'figures'/('05_vision_per_task.'+ext),dpi=160)
    plt.close(fig)
attention_splits=[]
for label,lo,hi in [('development_05_09',5,9),('extension_10_19',10,19)]:
    rs=[r for r in pairs if r['case']=='language-w2-g64-no-clip-attention' and lo<=r['init_state_index']<=hi]
    if not rs:continue
    assert len(rs)==10*(hi-lo+1),'incomplete attention split'
    attention_splits.append(dict(split=label,initial_state_start=lo,initial_state_end=hi,
        episodes=len(rs),successes=sum(r['candidate_success'] for r in rs),
        bf16_successes=sum(r['bf16_success'] for r in rs),
        source_shards=sorted(set(r['source_shard'] for r in rs)),**paired_stats(rs)))
(HERE/'data/language_attention_splits.json').write_text(json.dumps(attention_splits,indent=2)+'\n')
if attention_splits:
    fig,ax=plt.subplots(figsize=(7,4));x=np.arange(len(attention_splits))
    ax.bar(x-.18,[r['bf16_successes']/r['episodes'] for r in attention_splits],.36,label='Matched BF16')
    ax.bar(x+.18,[r['successes']/r['episodes'] for r in attention_splits],.36,label='Language W2 attention no clip')
    for i,r in enumerate(attention_splits):ax.text(i+.18,r['successes']/r['episodes']+.02,str(r['successes'])+'/'+str(r['episodes']),ha='center')
    ax.set_xticks(x);ax.set_xticklabels(['Development 5-9' if r['split'].startswith('development') else 'Extension 10-19' for r in attention_splits])
    ax.set_ylim(0,1.1);ax.set_ylabel('Closed-loop success rate');ax.legend(loc='lower left',fontsize=8)
    ax.set_title('Separate strategy selection from extended-state validation');fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(HERE/'figures'/('08_attention_state_extension.'+ext),dpi=160)
    plt.close(fig)
print(json.dumps(summary,indent=2))
