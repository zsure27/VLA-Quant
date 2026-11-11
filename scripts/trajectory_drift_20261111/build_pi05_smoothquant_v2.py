"""Build fixed-alpha SQ8/SQ4 profiles, with separate finite/reload admission."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from qvla.infra.common import digest,write_json,now
from qvla.pi05_torch_backend_v2 import load_policy,BACKEND_ID
from qvla.pi05_smoothquant_v2 import smooth,apply,groups,official_functions

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
capture_result=json.loads((a.capture/'result.json').read_text())
assert capture_result['gate']=='PASS_PI05_414_CALIBRATION_REPLAY_STATS'
assert digest(a.capture/'statistics.pt')==capture_result['profile_sha256']
cache=torch.load(a.capture/'statistics.pt',map_location='cpu',weights_only=True)
assert cache['backend']==BACKEND_ID
selected={}
for x in cache['samples']:selected.setdefault(x['suite'],x)
assert len(selected)==4
def infer(policy,row):
    assert row['reset_index']==0 and row['role']=='calibration'
    assert digest(row['path'])==row['sha256']
    with np.load(row['path'],allow_pickle=False) as q:
        obs={'observation/image':q['image'],'observation/wrist_image':q['wrist_image'],
             'observation/state':q['state'],'prompt':str(q['prompt'].item())}
        noise=q['policy_noise'];reference=q['raw_policy_chunk']
    actual=np.asarray(policy.infer(obs,noise=noise)['actions'])
    if actual.shape!=reference.shape or not np.isfinite(actual).all():raise ValueError('Nonfinite/malformed calibration actions')
    return actual,reference

t0=time.monotonic();policy=load_policy(a.root);mods=dict(policy._model.named_modules())
for name,_ in groups(policy._model):
    norm=mods[name]
    if type(norm).__name__=='GemmaRMSNorm':
        values=list(norm.parameters());assert all(x.dtype==torch.float32 for x in values),'Runtime norm must match FP32 CPU proof'
scales=smooth(policy._model,cache['statistics'],alpha=.5);assert len(scales)==126
equivalence=[]
for row in selected.values():
    actual,reference=infer(policy,row)
    mse=float(np.mean((actual.astype('float64')-reference)**2)/max(np.mean(reference.astype('float64')**2),1e-12))
    equivalence.append(dict(sample_sha256=row['sha256'],relative_mse=mse,max_abs=float(np.max(np.abs(actual-reference)))))
    if mse>1e-3:raise RuntimeError('Unquantized SQ reparameterization failed registered relative-MSE<=1e-3 gate')
del policy,mods,norm,values;gc.collect();torch.cuda.empty_cache()
official,source=official_functions(a.root)
payload=dict(format='PI05_TORCH_SMOOTHQUANT_V1',backend=BACKEND_ID,alpha=.5,
             checkpoint_sha256=cache['checkpoint_sha256'],calibration_capture_sha256=capture_result['profile_sha256'],
             statistics={n:{'shape':v['shape'],'absmax':v['absmax'],'count':v['count']} for n,v in cache['statistics'].items()},
             source_sha256=digest(source),weight_bits=4,precision_gates={},calibration_expected_actions={},
             calibration_probe_samples=list(selected.values()),
             unquantized_equivalence=equivalence,implementation='Pinned SQ fake quantizers; explicit Gemma gamma=1+w / adaRMS scale-shift folding; no INT4 kernel',
             exceptions='Embedding, patch convolution, norms, conditioning/action I/O projections retain original mixed BF16/FP32')
errors={}
for bits in [8,4]:
    policy=None
    try:
        policy=load_policy(a.root);apply(policy._model,payload,a.root,bits,require_admission=False)
        expected=[torch.from_numpy(infer(policy,row)[0].copy()) for row in selected.values()]
        payload['calibration_expected_actions'][str(bits)]=expected
        provisional=a.output/('profile-stage-a'+str(bits)+'.pt');torch.save(payload,provisional)
        del policy;policy=None;gc.collect();torch.cuda.empty_cache()
        saved=torch.load(provisional,map_location='cpu',weights_only=True)
        policy=load_policy(a.root);apply(policy._model,saved,a.root,bits,require_admission=False)
        for row,target in zip(selected.values(),expected):
            actual,_=infer(policy,row)
            assert np.array_equal(actual,target.numpy()),'Serialized profile action reload differs'
        assert all(not p.requires_grad for p in policy._model.parameters())
        payload['precision_gates'][str(bits)]='PASS_RELOAD_FINITE'
    except Exception as error:
        payload['precision_gates'][str(bits)]='FAIL_NUMERICAL_OR_RELOAD'
        errors[str(bits)]=dict(type=type(error).__name__,message=str(error))
    finally:
        if policy is not None:del policy
        gc.collect();torch.cuda.empty_cache()
payload['gate']='PASS_PI05_SQ_PROFILE_ADMISSION_PARTIAL_OR_FULL'
torch.save(payload,a.output/'profile.pt')
result=dict(gate=payload['gate'],precision_gates=payload['precision_gates'],errors=errors,
            profile_sha256=digest(a.output/'profile.pt'),profile_bytes=(a.output/'profile.pt').stat().st_size,
            smoothing_groups=126,targets=414,unquantized_equivalence=equivalence,wall_seconds=time.monotonic()-t0,time_utc=now())
write_json(a.output/'result.json',result);print(json.dumps(result))
if not any(x=='PASS_RELOAD_FINITE' for x in payload['precision_gates'].values()):raise SystemExit(1)
