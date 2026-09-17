"""Export only measured scope/BF16 disagreements with original video paths."""
import argparse,csv,json,re
from pathlib import Path
from baseline_shard_analysis import parse_episodes
p=argparse.ArgumentParser();p.add_argument('--eval-root',type=Path,required=True)
p.add_argument('--case',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()

def videos(case):
    logs=list(case.glob('EVAL-*.txt'));assert len(logs)==1
    rows=parse_episodes(logs[0].read_text())
    paths=re.findall(r'Saved rollout MP4 at path ([^\r\n]+)',(case/'console.log').read_text())
    assert len(rows)==len(paths)
    return {(r['task_id'],r['init_state_index']):str((Path('/root')/path).resolve()) for r,path in zip(rows,paths)}

reference={}
for shard in sorted(a.eval_root.glob('awq-baseline-shard-*')):
    if not (shard/'paired-results.json').exists():continue
    v=videos(shard/'bf16');assert not (reference.keys() & v.keys());reference.update(v)
out=[];seen=set()
for shard in sorted(a.eval_root.glob('awq-scope-shard-*')):
    paired=shard/'scope-paired-results.json'
    if not paired.exists():continue
    d=json.loads(paired.read_text())
    if d['case']!=a.case:continue
    v=videos(shard/a.case)
    for row in d['paired_episodes']:
        key=row['task_id'],row['init_state_index'];assert key not in seen;seen.add(key)
        if row['candidate_success']==row['bf16_success']:continue
        out.append(dict(task_id=key[0],init_state_index=key[1],bf16_success=row['bf16_success'],
            candidate_success=row['candidate_success'],bf16_video=reference[key],candidate_video=v[key],source_shard=shard.name))
out.sort(key=lambda r:(r['task_id'],r['init_state_index']))
assert out,'No observed disagreements'
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print(json.dumps(dict(covered_episodes=len(seen),disagreement_pairs=len(out),output=str(a.output))))
