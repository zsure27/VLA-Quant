"""Audit complete initial15 cases only; retain all missing pair slots explicitly."""
import argparse,hashlib,json,sys,types
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.root.resolve();out=a.output.resolve();assert not out.exists();code=r/'code-cpu-prep-20261109-v1';sys.path[:0]=[str(code),str(r/'sources/openvla-oft'),str(r/'sources/LIBERO')]
 profiles={'awq_w4a16':r/'profiles/20261109-four-suite-reset0-v4/awq.pt','smoothquant_w4a8':r/'profiles/20261109-four-suite-reset0-v4/smoothquant.pt','smoothquant_w4a4':r/'profiles/20261109-four-suite-reset0-v4/smoothquant.pt','gptq_w4a16':r/'profiles/20261110-gptq-full-v1/profile.pt','omniquant_w4a16':r/'profiles/20261110-omni-lwc-full-v2/profile.pt'}
 source=code/'diagnostics/native_oft_trace_v1.py';body=source.read_text();old="profile = 'awq.pt' if m['condition'].startswith('awq') else 'smoothquant.pt'\n        assert result['profile_sha256'] == sha(Path('profiles/20261109-four-suite-reset0-v4')/profile)";assert body.count(old)==1;body=body.replace(old,"assert result['profile_sha256'] == sha(profile_paths[m['condition']])")
 module=types.ModuleType('native_initial15_audit');module.profile_paths=profiles;exec(compile(body,str(source)+'[profile path binding]','exec'),module.__dict__);hashes={}
 def digest(path):
  path=Path(path);st=path.stat();key=(str(path.resolve()),st.st_size,st.st_mtime_ns)
  if key not in hashes:
   h=hashlib.sha256()
   with path.open('rb') as stream:
    for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
   hashes[key]=h.hexdigest()
  return hashes[key]
 module.sha=digest;lookup={};sources=[]
 for name in ['20261110-053-oft-initial15-v1','20261110-053-oft-initial15-v2','20261110-053-oft-initial15-v3','20261110-053-oft-initial15-v4','20261110-053-oft-initial15-weight-v1']:
  file=r/'control'/name/'status.json'
  if not file.exists():continue
  d=json.loads(file.read_text());sources.append({'path':str(file),'sha256':digest(file),'status':d['status'],'complete_cases':len(d['completed'])})
  for row in d['completed']:
   case=row['case'];assert case['reset_index']==15;result=Path(row['result']);assert digest(result)==row['sha256'];key=(case['suite'],case['task_id'],case['condition']);assert key not in lookup,('Duplicate admitted case',key);lookup[key]=result.parent
 threshold_path=r/'analysis/20261109-oft-pilot-v1/frozen-thresholds.json';thresholds=json.loads(threshold_path.read_text());records=[];missing=[]
 from qvla.trajectory_diagnostics_v1 import sustained_windows
 for suite in module.SUITES:
  for task in range(10):
   for cond in profiles:
    if (suite,task,'bf16') not in lookup or (suite,task,cond) not in lookup:missing.append({'suite':suite,'task_id':task,'condition':cond,'reason':'missing complete BF16 or candidate; not filled as failure'});continue
    x=module.load(lookup[suite,task,cond]);ref=module.load(lookup[suite,task,'bf16']);t,rot,w=module.compare(x,ref);records.append({'suite':suite,'task_id':task,'condition':cond,'reset_index':15,'success':x['result']['success'],'reference_success':ref['result']['success'],'common_control_steps':len(t),'translation_p95_mm':float(np.percentile(t,95)*1000),'rotation_p95_deg':float(np.percentile(rot,95)),'translation_windows':sustained_windows(t,thresholds['translation_m'],thresholds['persistence']),'rotation_windows':sustained_windows(rot,thresholds['rotation_deg'],thresholds['persistence']),'first_same_input_sha256':x['first_input_sha'],'wall_s_including_load':x['result']['wall_s_including_load']})
 out.mkdir(parents=True);(out/'summary.json').write_text(json.dumps({'gate':'PASS_AVAILABLE_INITIAL15_RAW_QUERY_ACTION_POSE_VIDEO_AUDIT','planned_pairs':200,'available_pairs':len(records),'missing':missing,'metrics':records,'scope':'OFT reset15 only; partial coverage retained; not five initial states or cross-model evidence','frozen_threshold_sha256':digest(threshold_path),'frozen_audit_source_sha256':digest(source),'control_snapshots':sources},indent=2));(out/'raw-audit.json').write_text(json.dumps(module.AUDITS,indent=2));print('AUDITED_PAIRS',len(records),'MISSING',len(missing))

if __name__=='__main__':main()
