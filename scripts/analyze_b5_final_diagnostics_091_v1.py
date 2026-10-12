"""Posthoc diagnostics for completed 450 episodes; never selection inputs."""
import hashlib,json
from collections import Counter
from pathlib import Path
BASE=Path('/root/autodl-tmp/qvla-repro/backups/experiments/p5-contextual-routing/20261012-091-b5-pilot-v2')
path=BASE/'combined450-paired-analysis.json';combined=json.loads(path.read_text())
assert combined['gate']=='PASS_AUDITED_DISJOINT_B5_PAIRED_COMBINED' and combined['episodes']==450
rows=combined['rows'];policies={};identical=0;failures=[]
for phase in ('first50','reset-10-19','reset-20-29','reset-30-39','reset-40-49'):
 traces={}
 for policy in ('fixed-e0','fixed-e1','router','task-only'):
  q=[json.loads(x) for x in (BASE/phase/policy/'trace.jsonl').read_text().splitlines() if json.loads(x).get('event')=='query']
  traces[policy]=q;counts=Counter(r['expert'] for r in q)
  value=policies.setdefault(policy,{'queries':0,'expert0_queries':0,'expert1_queries':0,'episodes':0,'switches':0,'episodes_starting_e0':0})
  eps={}
  for r in q:eps.setdefault(r['episode'],[]).append(r)
  value['queries']+=len(q);value['expert0_queries']+=counts[0];value['expert1_queries']+=counts[1];value['episodes']+=len(eps)
  value['switches']+=sum(a['expert']!=b['expert'] for e in eps.values() for a,b in zip(e,e[1:]))
  value['episodes_starting_e0']+=sum(e[0]['expert']==0 for e in eps.values())
  if policy=='router':
   outcomes={f"{r['task_id']}:{r['init_state_index']}":r for r in rows}
   for key,e in eps.items():
    if not outcomes[key]['B5']:
     failures.append({'episode':key,'queries':len(e),'e0_indices':[r['query'] for r in e if r['expert']==0],'switches':sum(a['expert']!=b['expert'] for a,b in zip(e,e[1:])),'fixed_B3_success':outcomes[key]['B3'],'fixed_B4_success':outcomes[key]['B4']})
 def signature(q):
  return [(r['episode'],r['query'],r['expert'],r['observation_sha256'],r['processed_inputs'],r['raw_policy_chunk']) for r in q]
 identical+=int(signature(traces['fixed-e1'])==signature(traces['task-only']))
oracle=sum(r['B3'] or r['B4'] for r in rows);both_fail=[r for r in rows if not r['B3'] and not r['B4']]
value={'gate':'PASS_COMPLETED_B5_POSTHOC_DIAGNOSTICS','episodes':450,'combined_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'policies':policies,'task_only_fixed_B4_exact_query_trace_identical_phases':identical,'phase_count':5,'B5_failures':failures,'fixed_episode_oracle_successes':oracle,'best_fixed_successes':max(combined['summary'][n]['successes'] for n in ('B3','B4')),'B5_failures_where_a_fixed_succeeds':sum((r['B3'] or r['B4']) and not r['B5'] for r in rows),'both_fixed_fail_episodes':len(both_fail),'B5_rescues_both_fixed_fail':sum(r['B5'] for r in both_fail),'limitations':['Posthoc complete outcome diagnostic; no model input or continuation decision','Fixed whole-episode oracle is not deployable or an absolute dynamic routing upper bound','Exact equality of observed task-only/B4 traces supports collapse on this development cohort only','Selection sequence is not causal proof; same early B3 then B4 pattern occurs in rescues and breaks','Same ten tasks, no unseen-task or blind-test claim']}
with (BASE/'final-routing-diagnostic.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps(value))
