"""Strict complete B5 phase audit, independent of success direction."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qvla.b5_contract import VERSION,sha,save,source_hashes,check_range
def audit(paths,offset,count):
    expected={(t,r) for t in range(10) for r in range(offset,offset+count)}; reference=None;report={};observation_first=None
    for path in paths:
        m=json.loads((path/'manifest.json').read_text());v=json.loads((path/'invocation.json').read_text())
        if m.get('version')!=VERSION or m.get('role')!='development_evaluation' or m['source_sha256']!=source_hashes() or sha(path/'trace.jsonl')!=m['trace_sha256'] or v['source_sha256']!=source_hashes():raise ValueError('Wrong source/trace/data role')
        if (v.get('role'),v.get('offset'),v.get('count'),v.get('policy'))!=('development_evaluation',offset,count,m['policy']):raise ValueError('Actual evaluation command role/range/policy differs')
        episodes=m['episodes'];keys={(e['manifest']['task_id'],e['manifest']['init_state_index']) for e in episodes}
        if len(episodes)!=len(expected) or keys!=expected or any(e['aborted'] for e in episodes):raise ValueError('Incomplete/aborted/duplicate phase')
        beginnings={};first={};queries={};endings={};observations={}
        for line in (path/'trace.jsonl').read_text().splitlines():
            r=json.loads(line)
            if r['event']=='episode_error':raise ValueError('Program error in rollout')
            if r['event']=='episode_begin':
                key=f"{r['manifest']['task_id']}:{r['manifest']['init_state_index']}"
                if key in beginnings:raise ValueError('Duplicate beginning')
                beginnings[key]=r['manifest'];queries[key]=[]
            elif r['event']=='query':
                q=queries[r['episode']]
                if r['query']!=len(q) or r['expert'] not in (0,1) or len(r['raw_policy_chunk'])!=8 or any(len(x)!=7 or any(not isinstance(v,(int,float)) or not __import__('math').isfinite(v) for v in x) for x in r['raw_policy_chunk']):raise ValueError('Invalid query/action semantic')
                if not q:
                    first[r['episode']]=(beginnings[r['episode']]['init_state_sha256'],r['processed_inputs'])
                    observations[r['episode']]=r['observation_sha256']
                q.append(r)
            elif r['event']=='episode_end':endings[r['episode_key']]=r
        if len(first)!=len(expected) or set(first)!=set(endings):raise ValueError('Missing actual first query/end')
        for key,e in endings.items():
            if e['queries']!=len(queries[key]) or e['executed_steps']+e['discarded_actions']!=8*e['queries']:raise ValueError('Action chunk lifecycle mismatch')
        if reference is None:reference=first
        elif reference!=first:raise ValueError('First processed observation/task/reset pairing mismatch')
        report[m['policy']]={'episodes':len(episodes),'successes':m['successes'],'manifest_sha256':sha(path/'manifest.json')}
        observation_first=observations
    if not {'fixed-e0','fixed-e1','router','task-only'}<=set(report):raise ValueError('Missing best fixed or task-only controls')
    return {'gate':'PASS_B5_PHASE_PROTOCOL','offset':offset,'count':count,'policies':report,'first_observation_sha256':observation_first,'source_sha256':source_hashes(),'criterion':'protocol only; no success-rate continuation selection'}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--references',type=Path,nargs=2,required=True);p.add_argument('--offset',type=int,required=True);p.add_argument('--count',type=int,required=True);p.add_argument('--output',type=Path);p.add_argument('--require',type=Path);a=p.parse_args()
    check_range('development_evaluation',a.offset,a.count);result=audit(a.runs,a.offset,a.count)
    from scripts.audit_extended_peft_eval import audit_case
    result['references']={}
    for name,path in zip(('BF16','A0'),a.references):
        r=audit_case(path,name,a.offset,a.count)
        first={f'{key[0]}:{key[1]}':r['first'][i] for i,key in enumerate(r['keys'])}
        if first!=result['first_observation_sha256']:raise ValueError('Baseline/B5 actual first observations differ')
        result['references'][name]={'successes':sum(r['success']),'files':r['sha256']}
    if a.require:
        if json.loads(a.require.read_text())!=result:raise ValueError('Prior phase bytes/gate changed')
    elif a.output:save(a.output,result)
    else:p.error('Need --output or --require')
if __name__=='__main__':main()
