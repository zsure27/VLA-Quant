"""Measured outcome complementarity; outcome oracle is not a trained router."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--paired-a',type=Path,required=True)
p.add_argument('--paired-b',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
datasets=[json.loads(f.read_text()) for f in (a.paired_a,a.paired_b)]
maps=[]
for d in datasets:
    rows=d['paired_episodes'];m={(r['task_id'],r['init_state_index']):r for r in rows}
    assert len(m)==len(rows);maps.append(m)
assert maps[0].keys()==maps[1].keys()
rs=[]
for key in sorted(maps[0]):
    x,y=maps[0][key],maps[1][key]
    for name in ('init_state_sha256','model_seed','env_seed','bf16_success'):
        assert x[name]==y[name],name
    rs.append(dict(task_id=key[0],init_state_index=key[1],a_success=x['candidate_success'],b_success=y['candidate_success']))

def stats(rows):
    return dict(episodes=len(rows),a_successes=sum(r['a_success'] for r in rows),
        b_successes=sum(r['b_success'] for r in rows),
        a_only=sum(r['a_success'] and not r['b_success'] for r in rows),
        b_only=sum(r['b_success'] and not r['a_success'] for r in rows),
        both_success=sum(r['a_success'] and r['b_success'] for r in rows),
        both_fail=sum(not r['a_success'] and not r['b_success'] for r in rows),
        outcome_oracle_successes=sum(r['a_success'] or r['b_success'] for r in rows))
result=dict(cases=[d['case'] for d in datasets],summary=stats(rs),
    source_files=[dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in (a.paired_a,a.paired_b)],
    per_task=[dict(task_id=t,**stats([r for r in rs if r['task_id']==t])) for t in sorted(set(r['task_id'] for r in rs))],
    paired_episodes=rs,note='Outcome oracle uses test outcomes; only a complementarity upper bound, not routing performance or training data.')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['summary']))
