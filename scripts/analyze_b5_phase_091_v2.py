"""Analyze a completed audited B5 cohort; no GPU, no outcome-based selection."""
import argparse,json,re,sys
from pathlib import Path
ROOT=Path('/root/autodl-tmp/qvla-repro/control/b5-pilot-091-20261012-v2/source');sys.path.insert(0,str(ROOT))
from qvla.paired_metrics import comparison
from qvla.b5_contract import sha,save
BASE=Path('/root/autodl-tmp/qvla-repro/backups/experiments/p5-contextual-routing/20261012-091-b5-pilot-v2')
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--offset',type=int,required=True);p.add_argument('--count',type=int,required=True);p.add_argument('--combine-files',nargs='*',default=[]);p.add_argument('--combined-name');a=p.parse_args()
 phase=BASE/a.phase;gatepath=BASE/(a.phase+'-gate.json');gate=json.loads(gatepath.read_text());assert gate['gate']=='PASS_B5_PHASE_PROTOCOL' and (gate['offset'],gate['count'])==(a.offset,a.count)
 outcomes={};keys={};summary={};failures={};cost={}
 names={'B3':'fixed-e0','B4':'fixed-e1','B5':'router','task-only':'task-only'}
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
  wrapper=BASE/'stage-records'/(a.phase+'-'+policy);inv=json.loads((wrapper/'invocation.json').read_text());assert (wrapper/'exit-code.txt').read_text().strip()=='0'
  cost[name]['wall_clock_seconds_setup_included']=(wrapper/'exit-code.txt').stat().st_mtime-inv['started_unix']
 for name in ('BF16','A0'):
  d=phase/name;logs=list(d.glob('EVAL-*.txt'));assert len(logs)==1;txt=logs[0].read_text()
  ms=[json.loads(x) for x in re.findall(r'EPISODE_MANIFEST (\{.*\})',txt)];ok=[x=='True' for x in re.findall(r'Success: (True|False)',txt)]
  assert len(ms)==len(ok)==10*a.count
  outcomes[name]={(m['task_id'],m['init_state_index']):v for m,v in zip(ms,ok)};keys[name]={(m['task_id'],m['init_state_index']):m for m in ms}
 assert all(k==keys['B3'] for k in keys.values())
 rows=[{'task_id':t,'init_state_index':r,**{name:outcomes[name][(t,r)] for name in outcomes}} for t,r in sorted(outcomes['B3'])]
 for name in outcomes:
  summary[name]={'successes':sum(outcomes[name].values()),'episodes':10*a.count,'success_by_task':{str(t):sum(v for (task,r),v in outcomes[name].items() if task==t) for t in range(10)}}
 pairs={f'{c}_vs_{b}':comparison(rows,c,b) for c,b in [('B5','B3'),('B5','B4'),('B5','task-only'),('B5','BF16'),('B4','BF16'),('A0','BF16')]}
 value={'gate':'PASS_AUDITED_B5_PHASE_PAIRED_ANALYSIS','phase':a.phase,'offset':a.offset,'count':a.count,'rows':rows,'summary':summary,'comparisons':pairs,'failures':failures,'cost':cost,'protocol_gate_sha256':sha(gatepath),'limitations':['Same ten-task development, registered disjoint reset slice, not unseen tasks or blind test','Micro reset5 overlaps and is excluded','Ten task clusters; bootstrap uncertainty not independent-episode population guarantee','Failure states/actions saved; B5 evaluator has no videos, physical failure mechanism unclassified','Wall time includes setup and tracing; GPU forward and selector latency not separately instrumented','Heterogeneous full bank cost; pseudoquantized weights do not establish INT2 deployment benefit'],'continuation':'Preregistered protocol/resource/quota gate; independent of success direction'}
 save(BASE/(a.phase+'-paired-analysis.json'),value);print(json.dumps({'summary':summary,'comparisons':pairs,'cost':cost}))
 if a.combine_files:
  assert a.combined_name and '/' not in a.combined_name
  allrows=[];identities=[]
  for name in a.combine_files+[a.phase+'-paired-analysis.json']:
   assert '/' not in name
   file=BASE/name;v=json.loads(file.read_text());assert v['gate'] in ('PASS_AUDITED_FIRST50_PAIRED_ANALYSIS','PASS_AUDITED_B5_PHASE_PAIRED_ANALYSIS');allrows.extend(v['rows']);identities.append({'file':name,'sha256':sha(file)})
  assert len({(r['task_id'],r['init_state_index']) for r in allrows})==len(allrows)
  models=('BF16','A0','B3','B4','B5','task-only')
  summary={n:{'successes':sum(r[n] for r in allrows),'episodes':len(allrows),'success_by_task':{str(t):sum(r[n] for r in allrows if r['task_id']==t) for t in range(10)}} for n in models}
  pairs={f'{c}_vs_{b}':comparison(allrows,c,b) for c,b in [('B5','B3'),('B5','B4'),('B5','task-only'),('B5','BF16'),('B4','BF16'),('A0','BF16')]}
  save(BASE/a.combined_name,{'gate':'PASS_AUDITED_DISJOINT_B5_PAIRED_COMBINED','episodes':len(allrows),'rows':allrows,'summary':summary,'comparisons':pairs,'inputs':identities,'random':'first50 diagnostic only; not extended or pooled as six-policy comparison','micro_excluded':True,'limitations':value['limitations']})
  print(json.dumps({'combined':summary,'comparisons':pairs}))
if __name__=='__main__':main()

