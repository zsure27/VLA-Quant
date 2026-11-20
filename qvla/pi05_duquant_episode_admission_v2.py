"""Private calibration probes, generated only after actual 414-layer admission."""
import base64,json
from pathlib import Path
import numpy as np
import torch
from qvla.infra.common import digest,write_json
from qvla.pi05_torch_backend_v2 import BACKEND_ID


def decode_expected(row):
    dtype=np.dtype(row['dtype']);shape=tuple(row['shape'])
    if dtype.kind!='f' or dtype.itemsize not in (4,8) or shape!=(10,7):raise ValueError('Wrong DuQuant action representation')
    raw=base64.b64decode(row['bytes_base64'],validate=True)
    if len(raw)!=70*dtype.itemsize:raise ValueError('Wrong DuQuant probe byte length')
    value=np.frombuffer(raw,dtype=dtype).reshape(shape).copy()
    if not np.isfinite(value).all():raise ValueError('Nonfinite admitted DuQuant action')
    return value


def load_probe(proof_path,profile_path,expected_identity):
    proof=json.loads(Path(proof_path).read_text());profile_path=Path(profile_path)
    if (proof.get('gate')!='PASS_DUQUANT_FULL_GPU_GATE_AND_PRIVATE_EPISODE_PROBES'
        or proof.get('backend')!=BACKEND_ID or proof.get('profile_sha256')!=digest(profile_path)
        or proof.get('model_identity')!=expected_identity
        or proof.get('full_gpu_gate_sha256')!=digest(profile_path.with_name('result.json'))):raise ValueError('DuQuant episode admission identity mismatch')
    rows=proof['samples']
    if len(rows)!=4 or {r['suite'] for r in rows}!={'libero_spatial','libero_object','libero_goal','libero_10'}:raise ValueError('Four calibration probes required')
    for row in rows:
        if row['role']!='calibration' or row['reset_index']!=0 or digest(row['path'])!=row['sha256']:raise ValueError('Wrong calibration input role/content')
        decode_expected(row['expected'])
    return rows


def prepare(profile_path,capture,output,model_identity):
    profile_path,capture,output=Path(profile_path),Path(capture),Path(output)
    if output.exists():raise FileExistsError('Frozen probe index already exists')
    gate=json.loads(profile_path.with_name('result.json').read_text())
    if gate['gate']!='PASS_PI05_ORIGINAL_DUQUANT_414_FP32_BINDING_RELOAD_FINITE_TRANSFORM' or gate.get('full_policy_admitted') is not True or gate.get('production_rotation_calibration') is not True or gate['profile_sha256']!=digest(profile_path):raise ValueError('Actual complete DuQuant GPU gate required')
    payload=torch.load(profile_path,map_location='cpu',weights_only=True)
    if payload['format']!='PI05_ORIGINAL_DUQUANT_NATIVE_FP32_BINDING_V2' or payload['backend']!=BACKEND_ID or len(payload['projections'])!=414:raise ValueError('Wrong DuQuant profile')
    if digest(capture/'statistics.pt')!=json.loads((capture/'result.json').read_text())['profile_sha256']:raise ValueError('Calibration cache changed')
    cache=torch.load(capture/'statistics.pt',map_location='cpu',weights_only=True)
    if [r['sha256'] for r in cache['samples']]!=payload['calibration_query_sha256']:raise ValueError('Calibration order changed')
    selected=[next(r for r in cache['samples'] if r['suite']==suite) for suite in ['libero_spatial','libero_object','libero_goal','libero_10']]
    with np.load(profile_path.with_name('calibration-actions.npz'),allow_pickle=False) as check:
        expected=check['quantized'].copy()
        if expected.shape!=(4,10,7) or not np.array_equal(expected,check['reloaded']):raise ValueError('Independent DuQuant reload actions differ')
    rows=[]
    for row,value in zip(selected,expected):
        if row['role']!='calibration' or row['reset_index']!=0 or digest(row['path'])!=row['sha256']:raise ValueError('Wrong calibration role/content')
        value=np.ascontiguousarray(value)
        rows.append(dict(suite=row['suite'],path=row['path'],sha256=row['sha256'],role='calibration',reset_index=0,
            expected=dict(dtype=value.dtype.str,shape=list(value.shape),bytes_base64=base64.b64encode(value.tobytes()).decode())))
    proof=dict(gate='PASS_DUQUANT_FULL_GPU_GATE_AND_PRIVATE_EPISODE_PROBES',backend=BACKEND_ID,model_identity=model_identity,
        profile_sha256=digest(profile_path),full_gpu_gate_sha256=digest(profile_path.with_name('result.json')),
        activation_bits=payload['activation_bits'],samples=rows,raw_calibration_actions_private=True)
    write_json(output,proof);load_probe(output,profile_path,model_identity)
    return dict(gate=proof['gate'],probes=4,profile_sha256=proof['profile_sha256'],admission_sha256=digest(output))
