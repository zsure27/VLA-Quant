"""Audit existing B3 training vs archived development inputs before any B4 GPU work."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qvla.split_scope import audit_scope, task_inventory, sha, INVENTORY
from scripts.audit_extended_peft_eval import audit_case
from qvla.on_policy_capture import observation_hash

REFERENCES = ('A4', 'BF16', 'A0')
PHASES = {'first50': (20, 5), 'next50': (25, 5), 'last100': (30, 10)}


def audit_materials(materials):
    inventory = task_inventory()
    manifest_path = Path(materials['student80']) / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    training = manifest['samples']
    if manifest.get('source_model_id') != 'A4' or len(training) != 80:
        raise ValueError('Require the accepted A4 student-state80 source')
    for row in training:
        if row['init_state_sha256'] != inventory[row['task_id']]['active_state_sha256'][row['init_state_index']]:
            raise ValueError('Training initial state differs from registered Spatial inventory')
        path=manifest_path.parent/row['file']
        if sha(path)!=row['sample_sha256']: raise ValueError('Training sample file changed')
        with np.load(path,allow_pickle=False) as sample:
            instruction=str(sample['instruction'].item())
            if instruction!=inventory[row['task_id']]['task_name'].replace('_',' '):
                raise ValueError('Training task ID/instruction mapping differs')
            if observation_hash(sample['image'],sample['wrist_image'],sample['state'],instruction)!=row['observation_sha256']:
                raise ValueError('Training observation content mismatch')
    files = {str(manifest_path): sha(manifest_path), str(INVENTORY): sha(INVENTORY)}
    files.update({str(manifest_path.parent/r['file']):r['sample_sha256'] for r in training})
    evaluation, paired, identity = [], [], {}
    for phase, (offset, count) in PHASES.items():
        item = materials['data_scope_reference_runs'][phase]
        gate_path = Path(item['gate'])
        gate = json.loads(gate_path.read_text(encoding='utf-8'))
        if not gate.get('gate', '').startswith('PASS_PROTOCOL'):
            raise ValueError('Archived reference lacks an accepted protocol gate')
        files[str(gate_path)] = sha(gate_path)
        results = {}
        for case in REFERENCES:
            directory = Path(item['cases'][case])
            result = audit_case(directory, case, offset, count)
            if result['sha256'] != gate['cases'][case]['sha256']:
                raise ValueError('Archived reference differs from its accepted hashes')
            results[case] = result
            logs = list(directory.glob('EVAL-*.txt'))
            for name in (logs[0], directory/'invocation.json', directory/'policy-queries.jsonl', directory/'on-policy-events.jsonl'):
                files[str(name)] = sha(name)
            # Preserve material fingerprints for subsequent contemporaneous pair matching.
            invocation = json.loads((directory/'invocation.json').read_text(encoding='utf-8'))
            identity[f'{phase}/{case}'] = invocation
        base = results['A4']
        if any(r['keys'] != base['keys'] or r['first'] != base['first'] for r in results.values()):
            raise ValueError('Archived reference policies are not strictly paired')
        directory = Path(item['cases']['A4'])
        log = next(directory.glob('EVAL-*.txt')).read_text(encoding='utf-8')
        manifests = [json.loads(m.group(1)) for m in re.finditer(r'EPISODE_MANIFEST (\{.*\})', log)]
        queries = [json.loads(line) for line in (directory/'on-policy-events.jsonl').read_text(encoding='utf-8').splitlines()]
        for serial, row in enumerate(manifests):
            task, reset = row['task_id'], row['init_state_index']
            if row['init_state_sha256'] != inventory[task]['active_state_sha256'][reset]:
                raise ValueError('Evaluation initial state differs from registered Spatial inventory')
            hashes = [q['observation_sha256'] for q in queries if q['record_type']=='query' and q['episode_serial']==serial]
            evaluation.append(dict(row, observation_sha256_all_queries=hashes))
            paired.append({'task_id': task, 'init_state_index': reset, 'key': list(base['keys'][serial]),
                           'first': base['first'][serial], **{c: results[c]['success'][serial] for c in REFERENCES}})
    result = audit_scope(training, evaluation)
    if len(paired) != 200: raise ValueError('Require the full 200-episode archived reference cohort')
    result.update({'source_sha256': files, 'training_manifest': str(manifest_path), 'reference_rows': paired,
                   'reference_invocations': identity,
                   'scope': 'Same-task development; references are archived, not new concurrent runs',
                   'limits': 'Checks training against archived A4 visited observations; future B3/B4 traces require their own overlap check'})
    return result


def verify_receipt(path):
    record = json.loads(Path(path).read_text(encoding='utf-8'))
    if record.get('gate') != 'PASS_DATA_SCOPE' or record.get('claim') != 'within_task_development':
        raise ValueError('Data-scope receipt missing; no GPU admission')
    for name, expected in record['source_sha256'].items():
        if sha(name) != expected: raise ValueError(f'Data-scope receipt source changed: {name}')
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--materials',type=Path); p.add_argument('--output',type=Path)
    p.add_argument('--verify-receipt',type=Path)
    a=p.parse_args()
    if a.verify_receipt:
        result=verify_receipt(a.verify_receipt)
    else:
        if not a.materials or not a.output: p.error('--materials and --output required')
        if a.output.exists(): raise ValueError('Never overwrite a data audit')
        result=audit_materials(json.loads(a.materials.read_text(encoding='utf-8')))
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('gate','claim','training_episodes','evaluation_episodes','shared_tasks')}))


if __name__=='__main__': main()
