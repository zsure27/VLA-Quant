"""Audit/reuse 480 completed cases while QVLA occupies GPU; no new inference.

Only artifact-path selection extends the frozen raw auditing/pose math.
"""
import argparse,hashlib,json,os,pathlib,sys,types
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();r=a.root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);os.environ['CUDA_VISIBLE_DEVICES']=''
legacy=r/'code-cpu-prep-20261109-v1';sys.path[:0]=[str(legacy),str(r/'sources/openvla-oft'),str(r/'sources/LIBERO')]
from qvla.infra.common import digest
parent=r/'control/20261112-061-oft-long15-resume-v3';series=json.loads((parent/'series.json').read_text());rows=list(series['inherited_completed'])
for f in parent.glob('libero*-cases.json'):
 for row in json.loads(f.read_text())['completed']:rows.append(dict(**row['case'],result_path=row['result_path'],result_sha256=row['result_sha256']))
lookup={}
for row in rows:
 key=(row['suite'],row['task_id'],row.get('reset_index',row.get('reset')),row['condition']);assert key not in lookup,key
 f=pathlib.Path(row['result_path']);assert digest(f)==row['result_sha256'];result=json.loads(f.read_text());assert result['initial_state_index']==key[2];lookup[key]=f.parent
assert len(lookup)==480 and {x[2] for x in lookup}=={5,15}
profiles={'awq_w4a16':r/'profiles/20261109-four-suite-reset0-v4/awq.pt','smoothquant_w4a8':r/'profiles/20261109-four-suite-reset0-v4/smoothquant.pt','smoothquant_w4a4':r/'profiles/20261109-four-suite-reset0-v4/smoothquant.pt','gptq_w4a16':r/'profiles/20261110-gptq-full-v1/profile.pt','omniquant_w4a16':r/'profiles/20261110-omni-lwc-full-v2/profile.pt'}
profile_hash={name:digest(path) for name,path in profiles.items()}
source=legacy/'diagnostics/native_oft_trace_v1.py';body=source.read_text();old="profile = 'awq.pt' if m['condition'].startswith('awq') else 'smoothquant.pt'\n        assert result['profile_sha256'] == sha(Path('profiles/20261109-four-suite-reset0-v4')/profile)";assert body.count(old)==1
body=body.replace(old,"assert result['profile_sha256'] == expected_profiles[m['condition']]");m=types.ModuleType('_frozen_oft_completed_pairs');m.expected_profiles=profile_hash;exec(compile(body,str(source)+'[artifact paths only]', 'exec'),m.__dict__)
hashcache={}
def sha(p):
 p=pathlib.Path(p);st=p.stat();key=(str(p),st.st_size,st.st_mtime_ns)
 if key not in hashcache:hashcache[key]=digest(p)
 return hashcache[key]
m.sha=sha;cache=out/'pair-cache';cache.mkdir();metrics=[]
for key,path in sorted(lookup.items()):
 suite,task,reset,condition=key
 if condition=='bf16':continue
 assert condition in profiles
 cand=m.load(path);ref=m.load(lookup[suite,task,reset,'bf16']);t,rot,w=m.compare(cand,ref);n=len(t);v=(cand['position'][:n]-ref['position'][:n])*1000
 name=f'{suite}-task{task}-reset{reset}-{condition}.npz';file=cache/name
 np.savez_compressed(file,translation_mm=t*1000,rotation_deg=rot,translation_vector_mm=v,rotation_vector_deg=w*180/np.pi)
 row=dict(suite=suite,task_id=task,reset_index=reset,condition=condition,success=cand['result']['success'],reference_success=ref['result']['success'],common_steps=n,translation_p95_mm=float(np.percentile(t*1000,95)),rotation_p95_deg=float(np.percentile(rot,95)),first_same_input_chunk_mse=float(np.mean((cand['first_chunk']-ref['first_chunk'])**2)),cache=name,cache_sha256=digest(file),candidate=str(path),reference=str(lookup[suite,task,reset,'bf16']))
 metrics.append(row);m.CACHE.pop(str(path),None)
assert len(metrics)==400
result=dict(gate='PASS_480_COMPLETED_CASES_400_PAIRED_CPU_CACHE',scope='4 suites x10 tasks x2 initial states; development only; no new rollout, no active QVLA read',baseline_cases=80,quantized_pairs=400,profile_sha256=profile_hash,frozen_raw_math_sha256=digest(source),cache_math='original native load/compare; artifact identity paths only extended; signed error vectors derived without padding or warping',metrics=metrics)
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');(out/'raw-audit.json').write_text(json.dumps(m.AUDITS,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['metrics','profile_sha256']}),flush=True)
