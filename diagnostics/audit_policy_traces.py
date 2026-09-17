"""Audit original query-time traces; no cross-trajectory error comparisons."""
import argparse,hashlib,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--raw-root',type=Path,action='append',required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args();results=[]
for root in a.raw_root:
    for path in sorted(root.glob('awq-scope-shard-*/*/policy-queries.jsonl')):
        shard=path.parent.parent
        assert (shard/'exit-code.txt').read_text().strip()=='0'
        count=0
        for line in path.open():
            r=json.loads(line);chunk=r['raw_policy_chunk'];state=r['state']
            assert r['finite'] is True and len(chunk)==8 and all(len(x)==7 for x in chunk)
            assert len(state)==8 and r['state_space']=='raw_proprio_before_get_action'
            assert all(math.isfinite(float(v)) for row in chunk for v in row)
            assert all(math.isfinite(float(v)) for v in state)
            count+=1
        assert count>0
        results.append(dict(source_shard=shard.name,case=path.parent.name,queries=count,
            all_actions_and_proprio_finite=True,path=path.as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
result=dict(files=results,total_queries=sum(r['queries'] for r in results),
    note='8x7 raw action chunks before gripper processing. Query counts are not episodes; trajectories are not matched states.')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(files=len(results),total_queries=result['total_queries'],all_finite=True)))
