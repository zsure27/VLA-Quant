"""Quantify constant-choice control trace equality after all runs are complete."""
import json
from pathlib import Path
BASE=Path('/root/autodl-tmp/qvla-repro/backups/experiments/p5-contextual-routing/20261012-091-b5-pilot-v2')
out={}
for phase in ('first50','reset-10-19','reset-20-29','reset-30-39','reset-40-49'):
 def read(policy):
  return {(r['episode'],r['query']):r for r in (json.loads(x) for x in (BASE/phase/policy/'trace.jsonl').read_text().splitlines()) if r.get('event')=='query'}
 a=read('fixed-e1');b=read('task-only');assert a.keys()==b.keys()
 obs=[];processed=[];actions=[];maximum=0.0
 for key in a:
  if a[key]['observation_sha256']!=b[key]['observation_sha256']:obs.append(key)
  if a[key]['processed_inputs']!=b[key]['processed_inputs']:processed.append(key)
  error=max(abs(x-y) for ar,br in zip(a[key]['raw_policy_chunk'],b[key]['raw_policy_chunk']) for x,y in zip(ar,br))
  maximum=max(maximum,error)
  if error:actions.append(key)
 out[phase]={'queries':len(a),'observation_differences':len(obs),'processed_input_differences':len(processed),'action_differences':len(actions),'max_abs_raw_action_difference':maximum,'first_observation_difference':obs[:1],'first_processed_difference':processed[:1],'first_action_difference':actions[:1]}
with (BASE/'taskonly-trace-comparison.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
