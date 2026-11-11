"""Original GPTQ kernel on bounded, same-BF16 calibration rows; per-layer shards.

Recipe: group128 (ragged last group), damp0.01, block128, no act order.
BF16 activation reservoirs stay fixed; this is not sequential block propagation
or a reproduction of a paper's published calibration budget.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import types

import numpy as np
import torch

from qvla.infra.common import digest,write_json,now
from qvla.infra.lifecycle import quota_decision
from qvla.pi05_torch_backend_v2 import load_policy,target_names,BACKEND_ID
from qvla.pi05_packed_weight_v1 import pack,unpack,weight_sha

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--capture',type=Path,required=True);p.add_argument('--round-dir',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);(a.output/'layers').mkdir()
source=a.root/'sources/gptq-archive-v1'
assert digest(source/'gptq.py')=='bb47468cc2c788577562751de414d76eeb9968ff1c6098e21153557ef3d154c7'
assert digest(source/'quant.py')=='211528568b962b868d20fc58b55200b615f41ebd9f36b4d2d7e4a196a095ea5a'
baseline_flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
sys.path.insert(0,str(source));import gptq
torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=baseline_flags
capture_result=json.loads((a.capture/'result.json').read_text())
assert capture_result['gate']=='PASS_PI05_414_CALIBRATION_REPLAY_STATS'
assert digest(a.capture/'statistics.pt')==capture_result['profile_sha256']
cache=torch.load(a.capture/'statistics.pt',map_location='cpu',weights_only=True)
assert cache['backend']==BACKEND_ID
control=json.loads((a.round_dir/'config.json').read_text())['control']

def quantize_layers(policy):
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    modules=dict(policy._model.named_modules());names=target_names(policy._model);assert len(names)==414
    entries=[]
    for index,name in enumerate(names):
        if quota_decision(json.loads((a.round_dir/'quota.json').read_text()),control)=='CLOSE':
            raise RuntimeError('Quota closure; completed layer shards retained')
        module=modules[name];row=cache['statistics'][name];assert list(module.weight.shape)==row['shape']
        # Bound the known largest Hessian temporaries before allocating them.
        needed=module.in_features**2*4*7+512*1024**2
        free,_=torch.cuda.mem_get_info()
        if needed>free:raise RuntimeError('Insufficient measured VRAM for GPTQ Hessian '+name)
        engine=gptq.GPTQ(module);engine.quantizer=gptq.Quantizer()
        engine.quantizer.configure(4,perchannel=True,sym=False,mse=False)
        scales=[];zeros=[];original=engine.quantizer.find_params
        def capture_find(self,value,weight=False):
            original(value,weight=weight)
            if value.shape[-1]<=128:
                scales.append(self.scale.detach().clone());zeros.append(self.zero.detach().clone())
        engine.quantizer.find_params=types.MethodType(capture_find,engine.quantizer)
        x=row['rows'].to(device=module.weight.device,dtype=module.weight.dtype).unsqueeze(0)
        engine.add_batch(x,None)
        started=time.monotonic();engine.fasterquant(blocksize=128,percdamp=.01,groupsize=128,actorder=False,static_groups=False)
        assert len(scales)==math.ceil(module.in_features/128)
        assert torch.isfinite(module.weight).all()
        packed=pack(module.weight,scales,zeros,128)
        file=a.output/'layers'/('%04d.pt'%index);torch.save(dict(name=name,entry=packed),file)
        entries.append(dict(name=name,file='layers/'+file.name,sha256=digest(file),bytes=file.stat().st_size,
                            layer_wall_seconds=time.monotonic()-started,weight_sha256=weight_sha(module.weight)))
        write_json(a.output/'progress.json',dict(completed_layers=len(entries),target_layers=414,last_target=name,entries=entries,time_utc=now(),profile_admitted=False))
        engine.free();del engine,x,scales,zeros,packed;gc.collect();torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=baseline_flags
    return entries

started=time.monotonic();policy=load_policy(a.root);entries=quantize_layers(policy)
selected={}
for row in cache['samples']:selected.setdefault(row['suite'],row)
def actions(policy,row):
    assert row['role']=='calibration' and row['reset_index']==0 and digest(row['path'])==row['sha256']
    with np.load(row['path'],allow_pickle=False) as q:
        obs={'observation/image':q['image'],'observation/wrist_image':q['wrist_image'],
             'observation/state':q['state'],'prompt':str(q['prompt'].item())}
        noise=q['policy_noise']
    value=np.asarray(policy.infer(obs,noise=noise)['actions'])
    assert value.shape==(10,7) and np.isfinite(value).all()
    return value
expected=[torch.from_numpy(actions(policy,row).copy()) for row in selected.values()]
del policy;gc.collect();torch.cuda.empty_cache()
policy=load_policy(a.root);modules=dict(policy._model.named_modules())
for row in entries:
    assert digest(a.output/row['file'])==row['sha256']
    saved=torch.load(a.output/row['file'],map_location='cpu',weights_only=True)
    module=modules[row['name']];weight=unpack(saved['entry'],module.weight.device,module.weight.dtype)
    assert weight_sha(weight)==row['weight_sha256']
    with torch.no_grad():module.weight.copy_(weight)
for sample,target in zip(selected.values(),expected):assert np.array_equal(actions(policy,sample),target.numpy())
payload=dict(format='PI05_ORIGINAL_GPTQ_PACK_V1',backend=BACKEND_ID,checkpoint_sha256=cache['checkpoint_sha256'],
             capture_sha256=capture_result['profile_sha256'],entries=entries,calibration_probe_samples=list(selected.values()),
             calibration_expected_actions={'16':expected},precision_gates={'16':'PASS_RELOAD_FINITE'},
             recipe='Original GPTQ per-layer; BF16 reservoir128 rows; groups128 ragged, damp0.01, block128, actorderFalse; no sequential propagation',
             gate='PASS_PI05_GPTQ_414_PACK_RELOAD')
torch.save(payload,a.output/'profile.pt')
write_json(a.output/'result.json',dict(gate=payload['gate'],targets=414,precision_gates=payload['precision_gates'],
           profile_sha256=digest(a.output/'profile.pt'),packed_layer_bytes=sum(x['bytes'] for x in entries),
           wall_seconds=time.monotonic()-started,time_utc=now()))
print('PASS_PI05_GPTQ_414_PACK_RELOAD')
