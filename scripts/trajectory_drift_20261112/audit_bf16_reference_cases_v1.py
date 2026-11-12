"""Audit new BF16 references without inventing a candidate comparison."""
import argparse,json,pathlib,sys
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--round-dir',type=pathlib.Path,required=True);a=p.parse_args();r=a.root.resolve();rd=a.round_dir;sys.path[:0]=[str(r/'code-cpu-prep-20261109-v1'),str(r/'sources/openvla-oft'),str(r/'sources/LIBERO')]
from qvla.infra.common import digest
from diagnostics import native_oft_trace_v1 as m
series=json.loads((rd/'series.json').read_text());rows=[]
for f in rd.glob('libero*-cases.json'):rows+=json.loads(f.read_text())['completed']
assert len({(x['case']['suite'],x['case']['task_id'],x['case']['reset_index']) for x in rows})==len(rows)
partial=len(rows)!=len(series['cases'])
if partial:assert (pathlib.Path(series['root_control'])/'STOP_NEW_GPU').exists()
for row in rows:
 f=pathlib.Path(row['result_path']);assert digest(f)==row['result_sha256'];z=m.load(f.parent);assert z['result']['condition']=='bf16' and z['result']['initial_state_sha256']==row['case']['initial_state_sha256'] and z['result']['checkpoint_receipt_sha256']==series['checkpoint_receipt_sha256'] and z['result']['oft_source_commit']==series['expected_oft_source_commit']
result=dict(gate='PASS_AVAILABLE_INITIAL25_BF16_REFERENCES_QUOTA_STOP' if partial else 'PASS_INITIAL25_BF16_RAW_REFERENCE_CASES',planned=len(series['cases']),completed=len(rows),success=sum(json.loads(pathlib.Path(x['result_path']).read_text())['success'] for x in rows),quantized_pairs=0,role='unpaired BF16 reference cases; do not compare120 references to80 candidate denominators',actual_native_env_seed=0,legacy_env_seed_label_retained=7,wall_seconds_including_load_gates_trace_video=sum(x['wall_seconds'] for x in rows),raw_audits=m.AUDITS)
out=rd/'reference-audit';out.mkdir(exist_ok=False);(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='raw_audits'}))
