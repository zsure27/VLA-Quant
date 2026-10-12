"""Independent B5 provenance/data contracts; archived loaders are unchanged."""
from __future__ import annotations
import json, hashlib
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
VERSION='b5-causal-frozen-bank-pilot-v1-20261012'
ROLES={'router_train':range(0,4),'router_dev':range(4,5),'development_evaluation':range(5,50)}
SOURCES=('qvla/b5_contract.py','qvla/b5_features.py','qvla/b5_route_state.py','qvla/b5_expert_bank.py','qvla/b5_runtime.py','qvla/run_eval_b5.py','scripts/train_b5_router.py','scripts/label_b5_observations.py','scripts/prepare_b5_router_data.py','scripts/smoke_b5_gpu.py','scripts/audit_b5_protocol.py','configs/experiments/b5_pilot_v1_20261012.json')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def source_hashes():return {p:sha(ROOT/p) for p in SOURCES}
def check_range(role,offset,count):
    if role not in ROLES or type(offset)!=int or type(count)!=int or count<1 or not set(range(offset,offset+count))<=set(ROLES[role]):raise ValueError('Unregistered B5 data role/reset range')
def candidate_contract(materials):
    from qvla.extended_peft import validate_artifact as old_b3
    from qvla.b4_joint_peft import validate_artifact as old_b4
    # Verify historical training identity under its historical admission range.
    # The independent check_range above admits NEW data, not this old receipt.
    records=[]
    for model,validate in (('B3',old_b3),('B4',old_b4)):
        prefix=model.lower()
        request=SimpleNamespace(model_id=model,extended_peft_manifest=Path(materials[prefix+'_artifact']),
            recovery_training_manifest=Path(materials['student80'])/'manifest.json',
            awq_recovery_lora_state=Path(materials[prefix+'_adapter']),awq_visual_lora_state=None,
            task_suite_name='libero_spatial',initial_state_offset=20,num_trials_per_task=1)
        records.append(validate(request))
    a,b=records
    if a['checkpoint_identity']!=b['checkpoint_identity'] or a['training_manifest_sha256']!=b['training_manifest_sha256']:
        raise ValueError('Expert base or training provenance differs')
    registry=json.loads((ROOT/'configs/model_registry_v1.json').read_text())
    for key,value in registry['profile_sha256'].items():
        if sha(materials[key])!=value:raise ValueError('A4 profile changed: '+key)
    return records
def require_baseline_cpu(materials):
    receipt=json.loads(Path(materials['baseline_cpu_receipt']).read_text())
    if receipt.get('gate')!='PASS_CPU_ACTUAL_PROFILE_CHECKPOINT_AND_ROW_QUANTIZATION' or receipt.get('targets_checked')!=422:
        raise ValueError('Actual baseline tensor audit is required')
def require_gpu_admission(materials):
    require_baseline_cpu(materials)
    receipt=json.loads(Path(materials['baseline_gpu_receipt']).read_text())
    if receipt.get('gate')!='PASS_GPU_BASELINE_INPUT_AND_QUANTIZATION' or receipt.get('checkpoint_sha256')!=candidate_contract(materials)[0]['checkpoint_identity']['sha256']:
        raise ValueError('Baseline GPU/input admission is pending or incompatible')
    if receipt.get('runtime_evaluator_sha256')!=sha(Path(materials['oft_root'])/'experiments/robot/libero/run_libero_eval.py'):
        raise ValueError('Baseline GPU gate belongs to another deployed evaluator')
def require_bank_smoke(materials):
    receipt=json.loads(Path(materials['bank_smoke_receipt']).read_text())
    if receipt.get('gate')!='PASS_GPU_B5_FIXED_EXPERT_PARITY_QUERY10' or receipt.get('source_sha256')!=source_hashes() or receipt.get('queries')!=10:
        raise ValueError('Real B5 bank query10 parity smoke is required')
def audit_capture(directory,role):
    import numpy as np
    from qvla.on_policy_capture import observation_hash
    from qvla.b5_features import normalized_proprio,RAW_STATE
    manifest=json.loads((Path(directory)/'manifest.json').read_text())
    if manifest.get('version')!=VERSION or manifest.get('role')!=role or manifest.get('state_space')!='raw_proprio_before_get_action':raise ValueError('Unregistered router data')
    rows=manifest['samples'];keys=set();episodes=set()
    for row in rows:
        check_range(role,row['init_state_index'],1)
        key=(row['collector_expert'],row['task_id'],row['init_state_index'],row['query_in_episode'])
        if row['collector_expert'] not in (0,1) or row['task_id'] not in range(10) or row['query_in_episode']<0 or key in keys:raise ValueError('Invalid/duplicate router observation')
        keys.add(key);episodes.add(key[:3])
        path=(Path(directory)/row['file']).resolve()
        if Path(directory).resolve() not in path.parents or sha(path)!=row['sha256']:raise ValueError('Router sample path/SHA mismatch')
        with np.load(path,allow_pickle=False) as s:
            normalized=normalized_proprio(s['state'],state_space=RAW_STATE,norm_stats=manifest['proprio_norm_stats'])
            fingerprint=observation_hash(s['router_image'],s['router_wrist_image'],normalized,str(s['instruction'].item()))
            if row.get('observation_sha256')!=fingerprint:raise ValueError('Actual router observation content hash mismatch')
    expected={(e,t,r) for e in (0,1) for t in range(10) for r in ROLES[role]}
    if episodes!=expected or len(rows)>len(expected)*5:raise ValueError('Incomplete or over-budget router episode coverage')
    return manifest
