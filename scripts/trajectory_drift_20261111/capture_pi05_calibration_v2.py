"""Capture bounded activation statistics from separate reset0 observations."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from qvla.infra.common import digest,write_json,now
from qvla.pi05_torch_backend_v2 import load_policy,target_names,BACKEND_ID

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--parent-round',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
admission=json.loads((a.parent_round/'admission/result.json').read_text())
assert admission['gate']=='PASS_PI05_TORCH_INDEPENDENT_BF16_RELOAD_INPUT'
samples=[]
for d in sorted((a.parent_round/'case-runs').iterdir()):
    result=json.loads((d/'result.json').read_text())
    if result['role']!='calibration':continue
    assert result['reset_index']==0 and result['backend']==BACKEND_ID
    assert result['gate']=='PASS_PI05_TORCH_BF16_EPISODE'
    queries=sorted(d.glob('query-*.npz'))
    # Fixed coverage of the observed calibration trajectory, without success filtering.
    indices=np.linspace(0,len(queries)-1,min(8,len(queries))).round().astype(int)
    assert len(set(indices))==len(indices)
    for i in indices:
        f=queries[i];assert digest(f)==result['query_files'][f.name]
        samples.append(dict(path=str(f),sha256=digest(f),suite=result['suite'],task_id=0,reset_index=0,
                            role='calibration',query_index=int(i),source_result_sha256=digest(d/'result.json')))
assert len({x['suite'] for x in samples})==4 and len(samples)<=32
write_json(a.output/'samples.json',dict(gate='PASS_PI05_DISTINCT_RESET0_CALIBRATION',samples=samples,
    development_resets=[5,15,25,35,45],training_samples=0,label_based_filter=False))
policy=load_policy(a.root);names=target_names(policy._model);assert len(names)==414
modules=dict(policy._model.named_modules());states={};handles=[]
for index,name in enumerate(names):
    states[name]=dict(shape=list(modules[name].weight.shape),count=0,rows=None,keys=None,
                      absmax=None,sum_abs=None,rng=np.random.default_rng(7+index))
    def capture(module,args,name=name):
        x=args[0].detach().reshape(-1,module.in_features)
        if not torch.isfinite(x).all():raise ValueError('Nonfinite activation '+name)
        state=states[name];maximum=x.float().abs().amax(dim=0).cpu();total=x.float().abs().sum(dim=0).cpu()
        state['absmax']=maximum if state['absmax'] is None else torch.maximum(state['absmax'],maximum)
        state['sum_abs']=total if state['sum_abs'] is None else state['sum_abs']+total
        state['count']+=len(x)
        ids=torch.linspace(0,len(x)-1,min(8,len(x)),device=x.device).round().long()
        rows=x[ids].cpu().clone();keys=state['rng'].random(len(rows))
        if state['rows'] is not None:rows=torch.cat([state['rows'],rows]);keys=np.concatenate([state['keys'],keys])
        keep=np.argsort(keys)[:128];state['rows']=rows[torch.from_numpy(keep)];state['keys']=keys[keep]
    handles.append(modules[name].register_forward_pre_hook(capture))
replays=[];t0=time.monotonic()
try:
    for row in samples:
        with np.load(row['path'],allow_pickle=False) as q:
            obs={'observation/image':q['image'],'observation/wrist_image':q['wrist_image'],
                 'observation/state':q['state'],'prompt':str(q['prompt'].item())}
            noise=q['policy_noise'];expected=q['raw_policy_chunk']
        actions=np.asarray(policy.infer(obs,noise=noise)['actions'])
        assert np.array_equal(actions,expected),'BF16 calibration replay differs'
        replays.append(dict(sample_sha256=row['sha256'],actions_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),exact_equal=True))
finally:
    for handle in handles:handle.remove()
for state in states.values():
    assert state['count']>0 and state['rows'] is not None and torch.isfinite(state['absmax']).all()
    state['mean_abs']=state.pop('sum_abs')/state['count'];state.pop('rng');state.pop('keys')
payload=dict(format='PI05_TORCH_SHARED_CALIBRATION_V1',backend=BACKEND_ID,
             checkpoint_sha256=admission['checkpoint_sha256'],statistics=states,samples=samples,
             input_reservoir='128 priority-selected rows per layer from 8 evenly spaced rows per invocation; seed7+sorted target index',
             max_statistics_use_all_rows=True,gate='PASS_PI05_414_CALIBRATION_REPLAY_STATS')
torch.save(payload,a.output/'statistics.pt')
write_json(a.output/'result.json',dict(gate=payload['gate'],target_count=414,sample_count=len(samples),
           calibration_only_reset0=True,all_bf16_replays_exact=True,replays=replays,
           profile_sha256=digest(a.output/'statistics.pt'),profile_bytes=(a.output/'statistics.pt').stat().st_size,
           time_utc=now(),capture_wall_seconds=time.monotonic()-t0))
print(json.dumps(dict(gate=payload['gate'],targets=414,samples=len(samples))))
