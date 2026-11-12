"""Serial QVLA cases against exact previously collected same-cohort references."""
import argparse,json,os,pathlib,subprocess,sys,time
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--round-dir',type=pathlib.Path,required=True);p.add_argument('--suite',required=True);a=p.parse_args()
from qvla.infra.common import digest,write_json,now
from qvla.infra.lifecycle import quota_decision
series=json.loads((a.round_dir/'series.json').read_text());control=json.loads((a.round_dir/'config.json').read_text())['control'];root_control=pathlib.Path(series['root_control']);state=dict(state='RUNNING',suite=a.suite,completed=[])
for case in series['cases']:
 if case['suite']!=a.suite:continue
 if (root_control/'STOP_NEW_GPU').exists() or quota_decision(json.loads((a.round_dir/'quota.json').read_text()),control)!='CONTINUE':
  state['state']='STOPPED_NO_NEW_GPU';write_json(a.round_dir/(a.suite+'-cases.json'),state);return_code=0;break
 output=a.round_dir/'case-runs'/case['id'];assert not output.exists()
 ref=pathlib.Path(case['reference_result']);assert digest(ref)==case['reference_sha256']
 cmd=[sys.executable,str(pathlib.Path(__file__).with_name('collect_oft_qvla_case_v1.py')),'--root',str(a.root),'--gates',series['gates'],'--admission',str(a.round_dir/'admission/result.json'),'--reference',str(ref.parent),'--suite',case['suite'],'--task-id',str(case['task_id']),'--reset',str(case['reset_index']),'--output',str(output),'--round-id',case['id']]
 state.update(current_case=case,updated_at=now());write_json(a.round_dir/(a.suite+'-cases.json'),state)
 started=time.monotonic()
 with (a.round_dir/'case-logs'/(case['id']+'.log')).open('xb') as log:
  proc=subprocess.Popen(cmd,cwd=a.root,env=os.environ,stdout=log,stderr=subprocess.STDOUT);state['policy_pid']=proc.pid;write_json(a.round_dir/(a.suite+'-cases.json'),state);code=proc.wait()
 if code:raise RuntimeError('QVLA case failed before final paired admission: '+case['id'])
 state['completed'].append(dict(case=case,result_path=str(output/'result.json'),result_sha256=digest(output/'result.json'),wall_seconds=time.monotonic()-started));write_json(a.round_dir/(a.suite+'-cases.json'),state)
else:state['state']='COMPLETE_PENDING_RAW_AUDIT'
write_json(a.round_dir/(a.suite+'-cases.json'),state)
