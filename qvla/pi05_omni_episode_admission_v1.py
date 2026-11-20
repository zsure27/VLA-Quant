"""CPU-only probe index produced only after full OMNI_LWC GPU admission.

Contains raw calibration actions: save in the private run, never public Git.
"""
import base64,json
from pathlib import Path
import numpy as np
import torch
from qvla.infra.common import digest,write_json
from qvla.pi05_torch_backend_v2 import BACKEND_ID


def decode_expected(row):
    dtype=np.dtype(row['dtype']);shape=tuple(row['shape'])
    if dtype.kind!='f' or dtype.itemsize not in (4,8) or shape!=(10,7):
        raise ValueError('Unknown admitted OMNI_LWC action representation')
    data=base64.b64decode(row['bytes_base64'],validate=True)
    if len(data)!=70*dtype.itemsize:raise ValueError('OMNI_LWC probe byte length differs')
    value=np.frombuffer(data,dtype=dtype).reshape(shape).copy()
    if not np.isfinite(value).all():raise ValueError('Nonfinite OMNI_LWC admitted probe')
    return value


def load_probe(proof_path,profile_path,expected_identity):
    proof_path=Path(proof_path);profile_path=Path(profile_path)
    proof=json.loads(proof_path.read_text())
    if (proof.get('gate')!='PASS_OMNI_LWC_FULL_GPU_GATE_AND_PRIVATE_EPISODE_PROBES' or proof.get('backend')!=BACKEND_ID
            or proof.get('profile_sha256')!=digest(profile_path) or proof.get('model_identity')!=expected_identity):
        raise ValueError('OMNI_LWC episode admission identity mismatch')
    rows=proof['samples']
    if len(rows)!=4 or {r['suite'] for r in rows}!={'libero_spatial','libero_object','libero_goal','libero_10'}:
        raise ValueError('OMNI_LWC four-suite calibration probes differ')
    for row in rows:
        if row.get('role')!='calibration' or row.get('reset_index')!=0 or digest(row['path'])!=row['sha256']:
            raise ValueError('OMNI_LWC admitted calibration input differs')
        decode_expected(row['expected'])
    return rows


def prepare(profile_path,capture,output):
    profile_path=Path(profile_path);capture=Path(capture);output=Path(output)
    if output.exists():raise FileExistsError('Do not overwrite frozen episode admission')
    gate=json.loads(profile_path.with_name('result.json').read_text())
    if gate.get('gate')!='PASS_PI05_OMNI_LWC_NATIVE_414_RELOAD_FINITE' or gate['profile_sha256']!=digest(profile_path):
        raise ValueError('Actual full-model OMNI_LWC gate required')
    payload=torch.load(profile_path,map_location='cpu',weights_only=True)
    if payload.get('full_policy_admitted') is not True or payload.get('gate')!=gate['gate'] or payload.get('backend')!=BACKEND_ID:
        raise ValueError('Partial OMNI_LWC profile cannot admit episodes')
    if digest(capture/'statistics.pt')!=payload['capture_sha256']:raise ValueError('Calibration cache identity changed')
    cache=torch.load(capture/'statistics.pt',map_location='cpu',weights_only=True)
    if [r['sha256'] for r in cache['samples']]!=payload['query_sha256']:raise ValueError('Calibration query order differs')
    selected=[next(r for r in cache['samples'] if r['suite']==suite) for suite in sorted({r['suite'] for r in cache['samples']})]
    with np.load(profile_path.with_name('calibration-action-checks.npz'),allow_pickle=False) as values:
        expected=values['quantized'].copy()
        if expected.shape!=(4,10,7) or not np.array_equal(expected,values['reloaded']):raise ValueError('OMNI_LWC full reload actions differ')
    rows=[]
    for row,value in zip(selected,expected):
        if row['role']!='calibration' or row['reset_index']!=0 or digest(row['path'])!=row['sha256']:raise ValueError('Wrong calibration role/identity')
        value=np.ascontiguousarray(value)
        rows.append(dict(suite=row['suite'],path=row['path'],sha256=row['sha256'],role='calibration',reset_index=0,expected=dict(dtype=value.dtype.str,shape=list(value.shape),bytes_base64=base64.b64encode(value.tobytes()).decode())))
    proof=dict(gate='PASS_OMNI_LWC_FULL_GPU_GATE_AND_PRIVATE_EPISODE_PROBES',backend=BACKEND_ID,model_identity=payload['model_identity'],profile_sha256=digest(profile_path),full_gpu_gate_sha256=digest(profile_path.with_name('result.json')),samples=rows,raw_calibration_actions_private=True)
    write_json(output,proof);load_probe(output,profile_path,payload['model_identity'])
    return dict(gate=proof['gate'],profile_sha256=proof['profile_sha256'],admission_sha256=digest(output),probes=4)
