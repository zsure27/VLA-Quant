"""Frozen G64 clip factorial on official states 5-9 only, not extended cohorts."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
here=Path(__file__).resolve().parent;root=here.parents[2]
cases=['language-w2-g64-clip','language-w2-g64-no-clip-mlp',
       'language-w2-g64-no-clip-attention','language-w2-g64-no-clip']
found={};contracts={}
for raw in ['107-baseline-continuation-20260917','107-awq-continuation-20260918']:
    for p in (root/'results'/raw).glob('awq-scope-shard-*/scope-paired-results.json'):
        d=json.loads(p.read_text());case=d['case']
        if case not in cases:continue
        rs=[r for r in d['paired_episodes'] if 5<=r['init_state_index']<10]
        if not rs:continue
        assert case not in found and len(rs)==50
        assert {(r['task_id'],r['init_state_index']) for r in rs}=={(t,i) for t in range(10) for i in range(5,10)}
        assert all(r['bf16_success'] and not r['candidate_episode_errors'] for r in rs)
        contracts[case]=(p.parent/'CONTRACT_SHA256SUMS.txt').read_text().splitlines()
        found[case]=dict(episodes=50,successes=sum(r['candidate_success'] for r in rs),source_shard=p.parent.name)
assert set(found)==set(cases)
base=contracts[cases[0]][2].split()[0]
for case in cases:
    assert contracts[case][:2]==contracts[cases[0]][:2]
    if case==cases[0]:continue
    shard=found[case]['source_shard']
    paths=[root/'results'/raw/shard/'profile-intervention.json' for raw in ['107-baseline-continuation-20260917','107-awq-continuation-20260918']]
    receipt=json.loads(next(p for p in paths if p.exists()).read_text())
    assert receipt['source_profile_sha256']==base
matrix=[[found[cases[0]]['successes'],found[cases[1]]['successes']],
        [found[cases[2]]['successes'],found[cases[3]]['successes']]]
(here/'data/language_clip_factorial.json').write_text(json.dumps(dict(group_size=64,source_profile_sha256=base,
    initial_state_indices=list(range(5,10)),cases=found,matrix=matrix,
    note='Language-only W2; vision BF16; identical original AWQ scales. Q/K weight clipping naturally absent in all four cases.'),indent=2)+'\n')
fig,ax=plt.subplots(figsize=(7,4.5));im=ax.imshow([[v/50 for v in row] for row in matrix],vmin=0,vmax=1,cmap='viridis')
ax.set_xticks([0,1]);ax.set_xticklabels(['MLP clip retained','MLP clip removed'])
ax.set_yticks([0,1]);ax.set_yticklabels(['V/O clip retained','V/O clip removed'])
for row in range(2):
    for col in range(2):ax.text(col,row,str(matrix[row][col])+'/50',ha='center',va='center',fontsize=18,color='white' if matrix[row][col]<25 else 'black')
ax.set_title('Language W2 G64: clipping factorial\nFixed native AWQ scale; official states 5-9')
fig.colorbar(im,ax=ax,label='Closed-loop success rate');fig.tight_layout()
for ext in ('png','svg'):fig.savefig(here/'figures'/('07_language_clip_factorial.'+ext),dpi=160)
plt.close(fig)
