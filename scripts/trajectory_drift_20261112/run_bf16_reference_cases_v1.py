"""Sequential unchanged native BF16 collector; no already-complete case is rerun."""
import argparse,json,os,pathlib,subprocess,sys,time
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--round-dir',type=pathlib.Path,required=True);p.add_argument('--suite',required=True);a=p.parse_args()
from qvla.infra.common import digest,write_json,now
from qvla.infra.lifecycle import quota_decision
series=json.loads((a.round_dir/'series.json').read_text());control=json.loads((a.round_dir/'config.json').read_text())['control'];root=pathlib.Path(series['root_control']);legacy=a.root/'code-cpu-prep-20261109-v1';state=dict(state='RUNNING',suite=a.suite,completed=[])
for case in series['cases']:
 if case['suite']!=a.suite:continue
 if (root/'STOP_NEW_GPU').exists() or quota_decision(json.loads((a.round_dir/'quota.json').read_text()),control)!='CONTINUE':state['state']='STOPPED_NO_NEW_GPU';write_json(a.round_dir/(a.suite+'-cases.json'),state);break
 for path,sha in series['external_frozen_sources'].items():assert digest(pathlib.Path(path))==sha
 output=a.round_dir/'case-runs'/case['id'];assert not output.exists();cmd=[sys.executable,str(legacy/'qvla/execute_oft_drift_v3.py'),'--oft-source',str(a.root/'sources/openvla-oft'),'--checkpoint',str(a.root/'models/openvla_oft'),'--suite',case['suite'],'--task-id',str(case['task_id']),'--initial-state-index',str(case['reset_index']),'--model-seed','7','--round-id',case['id'],'--output',str(output)]
 state.update(current_case=case,updated_at=now());write_json(a.round_dir/(a.suite+'-cases.json'),state);env=dict(os.environ,PYTHONPATH=str(legacy)+os.pathsep+os.environ.get('PYTHONPATH',''));start=time.monotonic()
 with (a.round_dir/'case-logs'/(case['id']+'.log')).open('xb') as f:
  child=subprocess.Popen(cmd,cwd=a.root,env=env,stdout=f,stderr=subprocess.STDOUT);state['policy_pid']=child.pid;write_json(a.round_dir/(a.suite+'-cases.json'),state);code=child.wait()
 if code:raise RuntimeError('Native BF16 reference failed: '+case['id'])
 result=output/'result.json';z=json.loads(result.read_text());assert z['gate']=='PASS_ACTUAL_OFT_BF16_NATURAL_TRACE' and z['initial_state_sha256']==case['initial_state_sha256'] and z['oft_source_commit']==series['expected_oft_source_commit']
 state['completed'].append(dict(case=case,result_path=str(result),result_sha256=digest(result),wall_seconds=time.monotonic()-start));write_json(a.round_dir/(a.suite+'-cases.json'),state)
else:state['state']='COMPLETE_PENDING_RAW_AUDIT'
write_json(a.round_dir/(a.suite+'-cases.json'),state)
