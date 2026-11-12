from pathlib import Path
import json,datetime
r=Path('/root/autodl-tmp/trajectory-drift-20261107');root=r/'control/20261112-061-oft-qvla-resume-v6';stages=[r/'control/20261112-061-oft-other-configs-v4/stages/qvla-pilot-v4',r/'control/20261112-061-oft-other-configs-v4/stages/qvla-complete80-v1',r/'control/20261112-061-oft-qvla-resume-v5/stages/qvla-remaining62-v2',root/'stages/qvla-remaining9-v1'];rows=[];groups=[]
for s in stages:
 cases=[]
 for f in s.glob('libero*-cases.json'):cases+=json.loads(f.read_text())['completed']
 rows+=cases;groups.append(dict(stage=str(s),completed=len(cases),wall_seconds=sum(x['wall_seconds'] for x in cases),original_query_npz=sum(len(list(Path(x['result_path']).parent.glob('episodes/*/query-*.npz'))) for x in cases)))
keys=[(x['case']['suite'],x['case']['task_id'],x['case']['reset_index']) for x in rows];assert len(rows)==len(set(keys))==80
out=dict(gate='PASS_QVLA80_COMPLETE_CASE_COSTS',complete_cases=80,original_query_npz=sum(x['original_query_npz'] for x in groups),wall_seconds=sum(x['wall_seconds'] for x in groups),cost_scope='sum of each complete-case subprocess wall time, includes loading/first-input bridge/profile verification/trace/video; standalone calibration/admission/CPU analysis and incomplete/control-failed attempts are separate and not zero cost',groups=groups,incomplete_attempts_retained_not_counted_as_model_failure=True,not_int4_kernel_speedup=True);(root/'QVLA_FULL80_COSTS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
