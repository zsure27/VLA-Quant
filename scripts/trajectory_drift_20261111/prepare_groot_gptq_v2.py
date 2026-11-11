"""Prepared GR00T GPTQ adapter; run only in a separately admitted GPU stage.

Fixed native-BF16 reservoirs (128 rows/layer), not sequential block propagation.
Original GPTQ: W4A16, group128, damp .01, block128, no act-order; nibble shards.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time
import types

import numpy as np
import torch

from qvla.groot_drift_20261111_v2 import load,infer,SUITES
from qvla.groot_sq_20261111_v2 import targets,read_obs
from qvla.infra.common import digest,write_json
from qvla.infra.lifecycle import quota_decision
from qvla.pi05_packed_weight_v1 import pack,unpack,weight_sha


def prepare(args):
    root,rd,suite,out=args.root,args.round_dir,args.suite,args.output
    sys.path[:0]=[str(root/'sources/Isaac-GR00T'),str(root/'sources/LIBERO')]
    native=rd/suite
    assert json.loads((native/'reference-gate.json').read_text())['gate']=='PASS_GROOT_REFERENCE_REPEAT_AND_CALIBRATION'
    out.mkdir(parents=True,exist_ok=False);(out/'layers').mkdir()
    source=root/'sources/gptq-upstream-v1'
    assert digest(source/'gptq.py')=='bb47468cc2c788577562751de414d76eeb9968ff1c6098e21153557ef3d154c7'
    assert digest(source/'quant.py')=='211528568b962b868d20fc58b55200b615f41ebd9f36b4d2d7e4a196a095ea5a'
    flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    sys.path.insert(0,str(source));import gptq
    torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags
    policy=load(root,suite);names=targets(root,suite,policy.model);modules=dict(policy.model.named_modules())
    files=sorted((native/'calibration-reset0').glob('query-*.npz'))
    selected=[files[i] for i in sorted(set(np.linspace(0,len(files)-1,min(8,len(files))).round().astype(int)))]
    rows={name:[] for name in names};hooks=[]
    # A fixed 128-row budget per target, 16 rows per selected observation.
    counts={name:0 for name in names}
    for name in names:
        def collect(_m,inputs,name=name):
            need=16-counts[name]
            if need<=0:return
            x=inputs[0].detach().flatten(0,-2)
            indices=torch.linspace(0,len(x)-1,min(need,len(x)),device=x.device).round().long()
            rows[name].append(x[indices].to('cpu',dtype=torch.bfloat16));counts[name]+=len(indices)
        hooks.append(modules[name].register_forward_pre_hook(collect))
    samples=[]
    for i,file in enumerate(selected):
        counts={name:0 for name in names};infer(policy,read_obs(file),i)
        samples.append(dict(path=str(file),sha256=digest(file),role='calibration',reset_index=0))
    for h in hooks:h.remove()
    if any(not value for value in rows.values()):raise ValueError('Unobserved GPTQ target')
    reservoirs={name:torch.cat(value) for name,value in rows.items()};del rows,hooks
    torch.save(dict(samples=samples,rows=reservoirs),out/'calibration-reservoirs.pt')
    control=json.loads((rd/'config.json').read_text())['control'];entries=[]
    try:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        for index,name in enumerate(names):
            if quota_decision(json.loads((rd/'quota.json').read_text()),control)=='CLOSE':raise RuntimeError('Quota closure')
            layer=modules[name];needed=layer.in_features**2*4*7+512*1024**2
            if torch.cuda.mem_get_info()[0]<needed:raise RuntimeError('Measured GPTQ Hessian memory insufficient: '+name)
            engine=gptq.GPTQ(layer);engine.quantizer=gptq.Quantizer()
            engine.quantizer.configure(4,perchannel=True,sym=False,mse=False)
            scales=[];zeros=[];original=engine.quantizer.find_params
            def find(self,value,weight=False):
                original(value,weight=weight)
                if value.shape[-1]<=128:scales.append(self.scale.detach().clone());zeros.append(self.zero.detach().clone())
            engine.quantizer.find_params=types.MethodType(find,engine.quantizer)
            x=reservoirs[name].to(layer.weight).unsqueeze(0);engine.add_batch(x,None)
            started=time.monotonic();engine.fasterquant(blocksize=128,percdamp=.01,groupsize=128,actorder=False,static_groups=False)
            assert len(scales)==math.ceil(layer.in_features/128)
            assert torch.isfinite(layer.weight).all()
            file=out/'layers'/f'{index:04d}.pt';torch.save(dict(name=name,entry=pack(layer.weight,scales,zeros,128)),file)
            entries.append(dict(name=name,file='layers/'+file.name,sha256=digest(file),weight_sha256=weight_sha(layer.weight),wall_seconds=time.monotonic()-started))
            write_json(out/'progress.json',dict(completed_layers=len(entries),target_layers=len(names),entries=entries,admitted=False))
            engine.free();del engine,x,scales,zeros;gc.collect();torch.cuda.empty_cache()
    finally:torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags
    first=read_obs(native/'bf16/query-00000.npz');expected,noise,_=infer(policy,first,0)
    del modules,policy;gc.collect();torch.cuda.empty_cache()
    policy=load(root,suite);modules=dict(policy.model.named_modules())
    for row in entries:
        file=out/row['file'];assert digest(file)==row['sha256']
        saved=torch.load(file,map_location='cpu',weights_only=True);layer=modules[row['name']]
        weight=unpack(saved['entry'],layer.weight.device,layer.weight.dtype)
        assert weight_sha(weight)==row['weight_sha256']
        with torch.no_grad():layer.weight.copy_(weight)
    actual,n,_=infer(policy,first,0)
    assert np.array_equal(expected,actual) and np.array_equal(noise,n)
    torch.save(dict(format='GROOT_ORIGINAL_GPTQ_W4A16_V1',suite=suite,entries=entries,samples=samples,
                    gate='PASS_GROOT_GPTQ_RELOAD_ACTION',recipe='native BF16 fixed reservoir128/layer; group128,damp.01,block128,no actorder; no sequential propagation'),out/'profile.pt')
    write_json(out/'result.json',dict(gate='PASS_GROOT_GPTQ_RELOAD_ACTION',profile_sha256=digest(out/'profile.pt'),targets=len(entries)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--round-dir',type=Path,required=True);p.add_argument('--suite',choices=SUITES,required=True)
    p.add_argument('--output',type=Path,required=True);prepare(p.parse_args())
