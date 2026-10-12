"""Analyze a completed audited B5 cohort; no GPU, no outcome-based selection."""
import json,re,sys
from pathlib import Path
ROOT=Path('/root/autodl-tmp/qvla-repro/control/b5-pilot-091-20261012-v2/source');sys.path.insert(0,str(ROOT))
from qvla.paired_metrics import comparison
from qvla.b5_contract import sha,save
BASE=Path('/root/autodl-tmp/qvla-repro/backups/experiments/p5-contextual-routing/20261012-091-b5-pilot-v2')
def main():
 phase=BASE/'first50';gate=json.loads((BASE/'first50-all-controls-gate.json').read_text());assert gate['gate']=='PASS_B5_PHASE_PROTOCOL'
 outcomes={};keys={};summary={};failures={};cost={}
 names={'B3':'fixed-e0','B4':'fixed-e1','B5':'router','task-only':'task-only','random':'random'}
 for name,policy in names.items():
  d=phase/policy;m=json.loads((d/'manifest.json').read_text());eps=m['episodes']
  outcomes[name]={(e['manifest']['task_id'],e['manifest']['init_state_index']):bool(e['success']) for e in eps}
  keys[name]={(e['manifest']['task_id'],e['manifest']['init_state_index']):e['manifest'] for e in eps}
  rows=[json.loads(x) for x in (d/'trace.jsonl').read_text().splitlines()];failed=[]
  for e in eps:
   if e['success']:continue
   k=e['episode_key'];qs=[r for r in rows if r.get('event')=='query' and r['episode']==k]
   failed.append({'episode':k,'queries':len(qs),'executed_steps':e['executed_steps'],'aborted':e['aborted'],'expert0_queries':sum(q['expert']==0 for q in qs),'expert1_queries':sum(q['expert']==1 for q in qs),'switches':sum(a['expert']!=b['expert'] for a,b in zip(qs,qs[1:])),'e0_query_indices':[q['query'] for q in qs if q['expert']==0]})
  failures[name]=failed;cost[name]={'queries':sum(e['queries'] for e in eps),'bank_parameters':m['bank_parameters'],'bank_tensor_bytes':m['bank_tensor_bytes'],'all_bank_cost_reported':True}
  wrapper=BASE/'stage-records'/('first50-'+policy);inv=json.loads((wrapper/'invocation.json').read_text());assert (wrapper/'exit-code.txt').read_text().strip()=='0'
  cost[name]['wall_clock_seconds_setup_included']=(wrapper/'exit-code.txt').stat().st_mtime-inv['started_unix']
 for name in ('BF16','A0'):
  d=phase/name;logs=list(d.glob('EVAL-*.txt'));assert len(logs)==1;txt=logs[0].read_text()
  ms=[json.loads(x) for x in re.findall(r'EPISODE_MANIFEST (\{.*\})',txt)];ok=[x=='True' for x in re.findall(r'Success: (True|False)',txt)]
  assert len(ms)==len(ok)==50
  outcomes[name]={(m['task_id'],m['init_state_index']):v for m,v in zip(ms,ok)};keys[name]={(m['task_id'],m['init_state_index']):m for m in ms}
 assert all(k==keys['B3'] for k in keys.values())
 rows=[{'task_id':t,'init_state_index':r,**{name:outcomes[name][(t,r)] for name in outcomes}} for t,r in sorted(outcomes['B3'])]
 for name in outcomes:
  summary[name]={'successes':sum(outcomes[name].values()),'episodes':50,'success_by_task':{str(t):sum(v for (task,r),v in outcomes[name].items() if task==t) for t in range(10)}}
 pairs={f'{c}_vs_{b}':comparison(rows,c,b) for c,b in [('B5','B3'),('B5','B4'),('B5','task-only'),('B5','random'),('B5','BF16'),('B4','BF16'),('A0','BF16')]}
 value={'gate':'PASS_AUDITED_FIRST50_PAIRED_ANALYSIS','phase':'first50','offset':5,'count':5,'rows':rows,'summary':summary,'comparisons':pairs,'failures':failures,'cost':cost,'protocol_gate_sha256':sha(BASE/'first50-all-controls-gate.json'),'limitations':['Same ten-task development reset5-9, not unseen tasks or blind test','Micro reset5 overlaps and is excluded','Ten task clusters; bootstrap uncertainty not independent-episode population guarantee','Failure states/actions saved; B5 evaluator has no videos, physical failure mechanism unclassified','Wall time includes setup and tracing; GPU forward and selector latency not separately instrumented','Heterogeneous full bank cost; pseudoquantized weights do not establish INT2 deployment benefit'],'continuation':'Preregistered protocol/resource/quota gate; independent of success direction'}
 save(BASE/'first50-paired-analysis.json',value);print(json.dumps({'summary':summary,'comparisons':pairs,'cost':cost}))
if __name__=='__main__':main()

