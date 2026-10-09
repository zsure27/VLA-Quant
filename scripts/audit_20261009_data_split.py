"""Reproduce the offline metadata audit; never opens holdout observations/labels."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from qvla.split_scope import sha,task_inventory,INVENTORY


def main():
    result_dir=ROOT/'results/experiments/p2-shared-peft/20261009-data-scope-audit'
    result_dir.mkdir(parents=True,exist_ok=True)
    paths={
        'inventory':INVENTORY,
        'offline_split':ROOT/'results/experiments/p1-data-contract/20260925-107-p1-data-contract/trajectory-split.json',
        'B0_B1_training':ROOT/'results/experiments/p2-shared-peft/20260929-107-p25-student-state80-distill/student-state80-manifest.json',
        'B3_training':ROOT/'results/experiments/p2-shared-peft/20261008-046-b2-b3/B3-student-state80-manifest.json',
        'prior_ptq_audit':ROOT/'results/experiments/p2-shared-peft/20261005-model-registry-audit/audit.json'}
    inventory=task_inventory();cohorts={}
    evaluation={h for t in inventory.values() for h in t['active_state_sha256'][20:40]}
    for kind in ('B0_B1_training','B3_training'):
        rows=json.loads(paths[kind].read_text())['samples']
        has_init=all('init_state_sha256' in r for r in rows)
        if has_init and not all(r['init_state_sha256']==inventory[r['task_id']]['active_state_sha256'][r['init_state_index']] for r in rows):
            raise ValueError('Inventory/content mismatch')
        cohorts[kind]={'observations':len(rows),'unique_observation_hashes':len({r['observation_sha256'] for r in rows}),
            'episodes':len({(r['task_id'],r['init_state_index']) for r in rows}),
            'tasks':sorted({r['task_id'] for r in rows}),'resets':sorted({r['init_state_index'] for r in rows}),
            'init_hash_against_inventory':'PASS' if has_init else 'MISSING_IN_ARCHIVED_MANIFEST',
            'init_hash_overlap_with_development20_39':len({r['init_state_sha256'] for r in rows}&evaluation) if has_init else None,
            'full_evaluation_observation_overlap':'PENDING_RAW_ARCHIVE_CHECK'}
    split=json.loads(paths['offline_split'].read_text())
    prior=json.loads(paths['prior_ptq_audit'].read_text())
    instructions={t['task_name'].replace('_',' '):k for k,t in inventory.items()}
    per_task={k:dict(Counter(r['role'] for r in split['episodes'] if instructions[r['instruction']]==k)) for k in inventory}
    known=prior['awq32_selected_trajectories']
    exposed=[r for r in known if r['later_role']!='peft_train']
    audit={'gate':'PASS_METADATA_AUDIT_NOT_FULL_DATA_CLEARANCE','primary_stage':'P2','GPU_started':False,
        'task_map':[{k:v for k,v in t.items() if k in ('task_id','task_name')} for t in inventory.values()],
        'training':cohorts,'evaluation_tasks':list(range(10)),'evaluation_resets':list(range(20,40)),
        'shared_task_count':10,'claim_allowed':'within-task development recovery',
        'offline_trajectories':len(split['episodes']),'offline_transitions':split['transition_count'],
        'offline_roles':dict(Counter(r['role'] for r in split['episodes'])),'offline_by_task':per_task,
        'known_historical_awq_calibration_exposure':exposed,
        'full_pipeline_final_holdout':'NOT_CERTIFIED: known PTQ exposure; other profiles/base checkpoint exclusions unresolved',
        'repair':'Versioned development/Recovery-task-transfer contracts; B4 GPU admission requires actual archived input-content audit',
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths.values()}}
    (result_dir/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    # Select tasks without consulting outcomes: first two SHA-ranked task names.
    import hashlib
    order=sorted(inventory,key=lambda t:hashlib.sha256(('recovery-task-transfer-v1|'+inventory[t]['task_name']).encode()).hexdigest())
    held=sorted(order[:2]);train=sorted(set(inventory)-set(held))
    source=json.loads(paths['B3_training'].read_text())['samples']
    planned={
        'version':'recovery-task-transfer-diagnostic-v1-20261009','status':'DESIGN_ONLY_NO_GPU_ADMISSION',
        'claim':'Tasks excluded from new Recovery training/SVD only; base OFT and frozen AWQ already exposed',
        'classification':'DIAGNOSTIC, previously inspected development tasks; not blind final evaluation',
        'selection':'Two smallest SHA256(recovery-task-transfer-v1|official_task_name); independent of outcomes',
        'recovery_training_tasks':train,'recovery_heldout_tasks':held,'training_resets':[0,1,2,3],
        'training_observations':64,'training_episodes':32,'teacher':'same-observation frozen BF16',
        'student_manifest_sha256':sha(paths['B3_training']),
        'training_sample_rows':[r for r in source if r['task_id'] in train],
        'excluded_sample_rows':[r for r in source if r['task_id'] in held],
        'response_svd_allowed_trajectory_ids':[r['trajectory_id'] for r in split['episodes'] if r['role']=='peft_train' and instructions[r['instruction']] in train],
        'initialization':'Recompute Response-SVD using train-task peft_train only; never reuse all-ten-task B3 initialization',
        'paired_models':['new B3 trained on 64 observations','new B4 trained on the identical 64 observations','A4'],
        'evaluation':'Two heldout Recovery tasks, dev reset20-39, 40 episodes/config; results by task',
        'runtime_work_pending':['versioned task-filtered evaluator','task-subset training contract and teacher cache','train-task-only SVD and smoke'],
        'no_final_role_reassignment':True}
    config=ROOT/'configs/data/recovery_task_transfer_v1_20261009.json';config.parent.mkdir(parents=True,exist_ok=True)
    config.write_text(json.dumps(planned,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'metadata_gate':audit['gate'],'same_tasks':10,'B3_init_overlap':cohorts['B3_training']['init_hash_overlap_with_development20_39'],
        'planned_train_tasks':train,'planned_heldout_tasks':held,'GPU_started':False}))


if __name__=='__main__':main()
