"""Record actual admissions and unresolved blockers; never launch experiments.

The disposition gate permits phase B under the registered plan. It explicitly
does not assert that blocked methods passed, or that the whole task completed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', type=Path, required=True)
    a = parser.parse_args()
    root = a.control.resolve()
    auth = load(root/'authorization.json')
    manifest = load(root/'task-manifest.json')
    assert auth['hostname'] == socket.gethostname()
    assert sha(root/'task-manifest.json') == auth['task_manifest_sha256']
    rows = []
    for bits in (8,4):
        stage = root/'stages'/f'20261120-084-duquant-a{bits}-pilot4-v1'
        state = load(stage/'status.json')
        assert len(state['stages']) == 4 and all(s.get('exit_code') == 0 for s in state['stages'])
        proof = load(stage/'FULL_RAW_AUDIT.json')
        assert proof['gate'] == 'PASS_DUQUANT_REGISTERED_CASES_FULL_RAW_AUDIT' and proof['complete'] == 4
        assert {(x['suite'],x['task_id'],x['reset_index']) for x in proof['rows']} == {(s,0,5) for s in manifest['suites']}
        for x in proof['rows']:
            assert sha(Path(x['result_path'])) == x['result_sha256']
        rows.append(dict(condition=f'duquant_w4a{bits}', status='ADMITTED_PILOT_RAW_AUDITED',
                         pilot_complete=4, full_trajectory_complete=False,
                         evidence=str(stage/'FULL_RAW_AUDIT.json'), evidence_sha256=sha(stage/'FULL_RAW_AUDIT.json')))
    readiness = root/'cpu/REMAINING_METHOD_READINESS_V1.json'
    other = load(readiness)
    for evidence in other['sources'].values():
        assert sha(Path(evidence['path'])) == evidence['sha256']
    graph = root/'stages/20261120-084-pi05-action-graph-prerequisite-v3'
    state = load(graph/'status.json')
    assert state['state'] not in ('RUNNING','PREFLIGHT')
    passed = all(x.get('exit_code') == 0 for x in state['stages']) and len(state['stages']) == 5
    if passed:
        f = graph/'action-graph/result.json'
        result = load(f)
        assert result['gate'] == 'PASS_PI05_NATIVE_TEN_STEP_ACTION_GRAPH_PREREQUISITE' and len(result['checks']) == 4
        assert result['qvla_full_admitted'] is False
    else:
        assert any(x.get('exit_code') not in (None,0) for x in state['stages'])
        f = graph/'model-admission.stdout.log'
        assert f.is_file()
    for item in other['rows']:
        row = dict(item, disposition='DOCUMENTED_BLOCKED_UNRESOLVED', full_method_admitted=False,
                   evidence=str(readiness), evidence_sha256=sha(readiness))
        if row['condition'] == 'qvla_w4a16':
            row.update(action_graph_prerequisite_passed=passed, graph_evidence=str(f), graph_evidence_sha256=sha(f),
                       reason='Native action-graph prerequisite '+('passed; ' if passed else 'did not pass; ')+
                       'physical-action sensitivity, channel allocation, quantized injection/reload and four-suite pilot remain unimplemented. No Hessian-proxy substitution.')
        rows.append(row)
    assert {x['condition'] for x in rows} == set(manifest['admission_order'])
    result = dict(gate='PASS_ALL_REQUESTED_CONFIGURATIONS_ADMITTED_OR_DOCUMENTED_BLOCKED',
                  round_id=root.name, task_manifest_sha256=auth['task_manifest_sha256'], rows=rows,
                  admitted_configurations=2, unresolved_configurations=5, all_requested_methods_admitted=False,
                  whole_task_complete=False, purpose='Disposition gate only; blocked work remains missing')
    path = root/'ADMISSION_PHASE_A_RESOLVED_GATE.json'
    with path.open('x') as out:
        json.dump(result,out,indent=2)
    print(json.dumps(dict(gate=result['gate'],admitted=2,blocked_unresolved=5,whole_task_complete=False,sha256=sha(path))))


if __name__ == '__main__':
    main()
