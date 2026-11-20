"""Audit a finished fixed stage, or explicitly interrupted whole-case prefix."""
import argparse
import importlib
import json
from pathlib import Path

from qvla.infra.common import digest, write_json, now


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--allow-interrupted-prefix', action='store_true')
    a = p.parse_args()
    stage = a.stage.resolve()
    state = json.loads((stage/'status.json').read_text())
    launch = json.loads((stage/'launch-receipt.json').read_text())
    process = Path('/proc',str(launch['supervisor_pid']),'cmdline')
    assert not process.exists() or str(stage).encode() not in process.read_bytes()
    series = json.loads((stage/'series.json').read_text())
    config = json.loads((stage/'config.json').read_text())
    method = config['quantization']['method']
    auditor = importlib.import_module({'gptq':'qvla.audit_pi05_torch_gptq_trace_v1',
                                      'smoothquant':'qvla.audit_pi05_torch_sq_trace_v2'}[method]).audit
    rows, missing, seen = [], [], set()
    for case in series['cases']:
        name = case['round_id']
        template = stage/'case-plans'/(name+'.json')
        assert digest(template) == case['template_sha256']
        actual = stage/'case-plans'/(name+'-runtime.json')
        if not actual.exists():
            missing.append(dict(round_id=name,reason='not_started'));continue
        plan = json.loads(actual.read_text())
        folder = Path(plan['output'])
        if not (folder/'result.json').exists():
            missing.append(dict(round_id=name,reason='no_complete_result',partial_files_preserved=True));continue
        proof = auditor(folder,actual)
        result = json.loads((folder/'result.json').read_text())
        key = (result['suite'],result['task_id'],result['reset_index'])
        assert key not in seen;seen.add(key)
        rows.append(dict(suite=key[0],task_id=key[1],reset_index=key[2],round_id=name,
                         success=bool(result['success']),result_path=str(folder/'result.json'),
                         result_sha256=digest(folder/'result.json'),runtime_sha256=digest(actual),audit=proof))
    if missing:
        assert a.allow_interrupted_prefix
    else:
        assert all(x.get('exit_code') == 0 for x in state['stages'])
    report = dict(gate='PASS_COMPLETE_CASE_PREFIX_RAW_AUDIT' if missing else 'PASS_REGISTERED_MATURE_STAGE_FULL_RAW_AUDIT',
                  condition=f'{method}_w4a'+str(config['quantization']['activation_bits']),
                  registered=len(series['cases']),complete=len(rows),successes=sum(x['success'] for x in rows),
                  whole_stage_complete=not missing,rows=rows,missing=missing,original_stage_state=state,
                  observed_at=now(),failed_or_partial_queries_not_added_to_denominator=True)
    dest = stage/'REGISTERED_RAW_AUDIT_V1.json'
    assert not dest.exists()
    write_json(dest,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['rows','original_stage_state','missing']}))


if __name__ == '__main__':
    main()
