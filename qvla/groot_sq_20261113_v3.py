"""Unfused SmoothQuant GR00T binding using pinned upstream quantization functions.

Each Linear uses x/s and W*s, with explicit runtime input scaling. This is
numerical fake quantization, not a packed INT4 kernel or a speedup claim.
"""
import argparse
import copy
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from qvla.groot_drift_20261113_v3 import load, infer, episode, SUITES
from qvla.infra.common import write_json, digest


def targets(root, suite, model):
    receipt = json.loads((root/'checks/20261109-008-cpu-prep-v1/CPU_RECEIPT.json').read_text())
    names = receipt['checkpoints'][suite]['upstream_script_selected_targets']
    modules = dict(model.named_modules())
    if len(names) != 116 or len(set(names)) != 116 or any(not isinstance(modules.get(n),torch.nn.Linear) for n in names):
        raise ValueError('GR00T target scope differs from original QuantVLA GR00T mapping')
    return names


def official(root):
    import importlib.util
    matches = [p for p in (root/'sources/quant-source-sparse-v2/smoothquant').rglob('fake_quant.py')
               if hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() ==
               'd0979f66c1bc38f4146db8365a162c5b356401f864636076395a978a9d569787']
    if len(matches) != 1: raise ValueError('Pinned upstream SmoothQuant source not unique')
    spec = importlib.util.spec_from_file_location('_groot_upstream_sq',matches[0])
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def read_obs(path):
    with np.load(path,allow_pickle=False) as data:
        return {k:(data[k].tolist() if k.startswith('annotation.') else data[k].copy())
                for k in data.files if k.startswith(('video.','state.','annotation.'))}


@torch.no_grad()
def apply(model, payload, quant, activation_bits=None):
    modules = dict(model.named_modules());handles=[]
    for name, scale_cpu in payload['scales'].items():
        layer = modules[name]
        if list(layer.weight.shape) != payload['shapes'][name]: raise ValueError('Weight shape differs')
        scale = scale_cpu.to(layer.weight)
        layer.weight.mul_(scale[None,:])
        if activation_bits is not None:
            weight = layer.weight.float().clone()
            quant.quantize_weight_per_channel_absmax(weight,n_bits=4)
            layer.weight.copy_(weight.to(layer.weight))
        def pre(_module,args,scale=scale,bits=activation_bits):
            value = (args[0]/scale).contiguous()
            if bits is not None:
                value = value.detach().clone()
                quant.quantize_activation_per_token_absmax(value,n_bits=bits)
            return (value,)+args[1:]
        handles.append(layer.register_forward_pre_hook(pre))
    return handles


def run(args):
    root,rd,suite=args.root,args.round_dir,args.suite
    sys.path[:0]=[str(root/'sources/Isaac-GR00T'),str(root/'sources/LIBERO')]
    parent=rd/suite
    references=json.loads(args.reference_map.read_text()) if args.reference_map else {}
    source=Path(references[suite]) if suite in references else parent
    parent.mkdir(exist_ok=True)
    if json.loads((source/'reference-gate.json').read_text())['gate']!='PASS_GROOT_REFERENCE_REPEAT_AND_CALIBRATION':
        raise ValueError('Missing completed native reference gate')
    out=parent/'smoothquant-profile';out.mkdir(exist_ok=False)
    files=sorted((source/'calibration-reset0').glob('query-*.npz'))
    selected=[files[i] for i in sorted(set(np.linspace(0,len(files)-1,min(8,len(files))).round().astype(int)))]
    obs=[read_obs(p) for p in selected]
    profile=dict(format='GROOT_SQ_UNFUSED_V1',suite=suite,alpha=0.5,scales={},shapes={},
                 calibration={p.name:digest(p) for p in selected},calibration_reset=0,
                 first_input_reference=str(source/'bf16/query-00000.npz'),weight_bits=4,
                 target_scope='116 original GR00T QuantVLA selected Linear targets',
                 runtime_input_scaling=True,real_integer_kernel=False)
    policy=load(root,suite);names=targets(root,suite,policy.model)
    modules=dict(policy.model.named_modules());stats={};hooks=[]
    for name in names:
        def collect(_m,inputs,name=name):
            x=inputs[0].detach().float().reshape(-1,inputs[0].shape[-1]).abs().amax(0)
            stats[name]=torch.maximum(stats[name],x) if name in stats else x
        hooks.append(modules[name].register_forward_pre_hook(collect))
    baseline=[]
    for i,value in enumerate(obs): baseline.append(infer(policy,value,i)[0])
    for h in hooks:h.remove()
    if set(stats)!=set(names): raise ValueError('Calibration did not reach every target')
    for name in names:
        layer=modules[name]
        w=layer.weight.detach().float().abs().amax(0).clamp_min(1e-5)
        s=(stats[name].clamp_min(1e-5).sqrt()/w.sqrt()).clamp(1e-5,1e5)
        if not torch.isfinite(s).all():raise ValueError('Nonfinite smoothing scale')
        profile['scales'][name]=s.cpu();profile['shapes'][name]=list(layer.weight.shape)
    del modules,stats,hooks
    quant=official(root)
    apply(policy.model,profile,quant)
    errors=[]
    for i,value in enumerate(obs):
        a=infer(policy,value,i)[0];b=baseline[i]
        errors.append(float(np.mean((a-b)**2)/max(float(np.mean(b*b)),1e-8)))
    write_json(out/'unquantized-equivalence.json',dict(gate='PASS' if max(errors)<=1e-3 else 'FAIL',
               per_sample_relative_action_mse=errors,preregistered_threshold=1e-3,scope=names))
    if max(errors)>1e-3:raise ValueError('Unquantized SQ reparameterization action gate failed')
    del policy
    gc.collect();torch.cuda.empty_cache()
    torch.save(profile,out/'profile.pt')
    restored=torch.load(out/'profile.pt',map_location='cpu',weights_only=True)
    assert restored['shapes']==profile['shapes'] and all(torch.equal(restored['scales'][n],profile['scales'][n]) for n in names)
    reference=source/'bf16'
    first=read_obs(reference/'query-00000.npz')
    for bits in [8,4]:
        policy=load(root,suite);apply(policy.model,restored,quant,bits)
        a,n,_=infer(policy,first,0)
        if not all(torch.isfinite(p).all() and not p.requires_grad for p in policy.model.parameters()):
            raise ValueError('Quantized weights nonfinite/unfrozen')
        del policy
        gc.collect();torch.cuda.empty_cache()
        policy=load(root,suite);apply(policy.model,restored,quant,bits)
        b,m,_=infer(policy,first,0)
        if not np.array_equal(a,b) or not np.array_equal(n,m):raise ValueError('SQ actual action reload failed')
        write_json(out/f'w4a{bits}-gate.json',dict(gate='PASS_GROOT_SQ_ACTUAL_RELOAD_FINITE',
                   target_count=len(names),profile_sha256=digest(out/'profile.pt'),activation_bits=bits))
        episode(policy,root,suite,parent/f'smoothquant-w4a{bits}',5,'development',reference=reference)
        del policy
        gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--round-dir',type=Path,required=True);p.add_argument('--reference-round',type=Path);p.add_argument('--reference-map',type=Path,required=True);p.add_argument('--suite',choices=SUITES,required=True)
    args=p.parse_args()
    try:run(args)
    except BaseException as error:
        write_json(args.round_dir/args.suite/'smoothquant-blocked.json',dict(gate='BLOCKED_GROOT_SQ',error=repr(error)))
        raise


if __name__=='__main__':main()
