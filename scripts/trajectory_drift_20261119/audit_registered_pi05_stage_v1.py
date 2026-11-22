"""Audit completed fixed cases with their original method-specific raw auditor."""
import argparse,importlib,json
from pathlib import Path
from qvla.infra.common import digest,write_json

p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--condition',choices=['bf16','smoothquant_w4a8','gptq_w4a16','awq_w4a16','omniquant_w4a16'],required=True);a=p.parse_args()
state=json.loads((a.stage/'status.json').read_text());assert state['state']=='PAUSED_SERVER_LEFT_ON' and all(x.get('exit_code')==0 for x in state['stages'])
series=json.loads((a.stage/'series.json').read_text())
module={'bf16':'qvla.audit_pi05_torch_trace_v2','smoothquant_w4a8':'qvla.audit_pi05_torch_sq_trace_v2','gptq_w4a16':'qvla.audit_pi05_torch_gptq_trace_v1','awq_w4a16':'qvla.audit_pi05_torch_awq_trace_v1','omniquant_w4a16':'qvla.audit_pi05_torch_omni_trace_v1'}[a.condition]
auditor=importlib.import_module(module).audit;rows=[];seen=set()
for case in series['cases']:
    key=(case['suite'],case['task_id'],case['reset_index']);assert key not in seen;seen.add(key)
    template=a.stage/'case-plans'/(case['round_id']+'.json');assert digest(template)==case['template_sha256']
    actual=a.stage/'case-plans'/(case['round_id']+'-runtime.json');plan=json.loads(actual.read_text())
    assert plan['condition']==a.condition and (plan['suite'],plan['task_id'],plan['reset_index'])==key
    folder=Path(plan['output']);proof=auditor(folder,actual);result=json.loads((folder/'result.json').read_text())
    assert result['backend']=='openpi-pytorch-original-independent-h10-v2' and result['role']=='development'
    rows.append(dict(suite=key[0],task_id=key[1],reset_index=key[2],success=bool(result['success']),result_sha256=digest(folder/'result.json'),runtime_sha256=digest(actual),audit=proof))
assert len(rows)==len(series['cases']) and rows
report=dict(gate='PASS_REGISTERED_PI05_CASES_ORIGINAL_RAW_AUDITOR',condition=a.condition,complete=len(rows),successes=sum(x['success'] for x in rows),rows=rows,science_unchanged=True)
destination=a.stage/'REGISTERED_RAW_AUDIT_V1.json'
if destination.exists():assert json.loads(destination.read_text())==report
else:write_json(destination,report)
print(json.dumps({k:report[k] for k in ['gate','condition','complete','successes']}))
