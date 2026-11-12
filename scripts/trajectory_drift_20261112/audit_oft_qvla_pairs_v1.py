"""CPU raw-query/action/pose/video paired audit using frozen OFT metric math."""
import argparse,json,pathlib,sys,types
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--round-dir',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();r,rd=a.root.resolve(),a.round_dir.resolve();a.output.mkdir(parents=True,exist_ok=False)
sys.path[:0]=[str(r/'code-cpu-prep-20261109-v1'),str(r/'sources/openvla-oft'),str(r/'sources/LIBERO')]
from qvla.infra.common import digest
source=r/'code-cpu-prep-20261109-v1/diagnostics/native_oft_trace_v1.py';body=source.read_text();old="profile = 'awq.pt' if m['condition'].startswith('awq') else 'smoothquant.pt'\n        assert result['profile_sha256'] == sha(Path('profiles/20261109-four-suite-reset0-v4')/profile)"
assert body.count(old)==1;body=body.replace(old,"assert result['profile_sha256'] == expected_gates_sha")
series=json.loads((rd/'series.json').read_text());m=types.ModuleType('_frozen_qvla_raw_auditor');m.expected_gates_sha=digest(pathlib.Path(series['gates']));exec(compile(body,str(source)+'[QVLA profile path only]','exec'),m.__dict__)
cache={}
def cached(p):
 p=pathlib.Path(p);k=(str(p),p.stat().st_size,p.stat().st_mtime_ns)
 if k not in cache:cache[k]=digest(p)
 return cache[k]
m.sha=cached;completed=[]
for f in rd.glob('libero*-cases.json'):completed+=json.loads(f.read_text())['completed']
assert len(completed)==len(series['cases']) and len({x['case']['id'] for x in completed})==len(completed)
metrics=[];curves=[]
for row in completed:
 case=row['case'];f=pathlib.Path(row['result_path']);assert digest(f)==row['result_sha256'];ref=pathlib.Path(case['reference_result']);assert digest(ref)==case['reference_sha256']
 c,b=m.load(f.parent),m.load(ref.parent);t,rot,w=m.compare(c,b)
 metrics.append(dict(suite=case['suite'],task_id=case['task_id'],reset_index=case['reset_index'],success=c['result']['success'],reference_success=b['result']['success'],common_steps=len(t),translation_p95_mm=float(np.percentile(t,95)*1000),rotation_p95_deg=float(np.percentile(rot,95)),first_same_input_chunk_mse=float(np.mean((c['first_chunk']-b['first_chunk'])**2))))
 curves.append(dict(suite=case['suite'],task_id=case['task_id'],reset_index=case['reset_index'],translation_mm=(t*1000).tolist(),rotation_deg=rot.tolist()))
result=dict(gate='PASS_QVLA_RAW_QUERY_ACTION_POSE_VIDEO_AND_PAIRED_CASES',planned=len(series['cases']),completed=len(completed),success=sum(x['success'] for x in metrics),reference_success=sum(x['reference_success'] for x in metrics),rescue=sum(x['success'] and not x['reference_success'] for x in metrics),break_count=sum(not x['success'] and x['reference_success'] for x in metrics),metrics=metrics,original_frozen_metric_source_sha256=digest(source),gates_sha256=m.expected_gates_sha)
(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');(a.output/'raw-audit.json').write_text(json.dumps(m.AUDITS,indent=2)+'\n');(a.output/'curves.json').write_text(json.dumps(curves)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='metrics'}))
