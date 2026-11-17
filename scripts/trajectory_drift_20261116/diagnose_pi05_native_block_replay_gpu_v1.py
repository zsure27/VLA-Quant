"""Real-policy calibration-only block replay gate; not an AWQ quantizer.

At a serial GPU boundary, check fixed calibration input/noise, unchanged actions,
and exact native vision/prefix/expert block replay before future quantizer work.
No development trajectory is consumed or included in a success denominator.
"""
import argparse
from pathlib import Path
import json
import inspect
import time
import numpy as np
import torch
from qvla.infra.common import digest,write_json,now
from qvla.pi05_torch_backend_v2 import load_policy,BACKEND_ID
from qvla.pi05_block_replay_v2 import capture_call,replay_call

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--capture',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();started=time.monotonic()
assert torch.cuda.is_available()
proof=json.loads((a.capture/'result.json').read_text())
assert proof['gate']=='PASS_PI05_414_CALIBRATION_REPLAY_STATS'
assert digest(a.capture/'statistics.pt')==proof['profile_sha256']
cache=torch.load(a.capture/'statistics.pt',map_location='cpu',weights_only=True)
assert cache['backend']==BACKEND_ID
samples={}
for row in cache['samples']:
    assert row['role']=='calibration' and row['reset_index']==0
    samples.setdefault(row['suite'],row)
assert len(samples)==4
a.output.mkdir(parents=True,exist_ok=False)
torch.cuda.reset_peak_memory_stats()
policy=load_policy(a.root)
assert torch.get_float32_matmul_precision()=='high' and torch.backends.cuda.matmul.allow_tf32
assert all(not x.requires_grad for x in policy._model.parameters())
versions={name:p._version for name,p in policy._model.named_parameters()}
modules={}
for name,module in policy._model.named_modules():
    cls=type(module).__name__
    kind='vision' if cls=='SiglipEncoderLayer' else ('expert' if 'gemma_expert' in name else 'prefix') if cls=='GemmaDecoderLayer' else None
    if kind is not None:modules.setdefault(kind,(name,module))
assert set(modules)=={'vision','prefix','expert'}

def action(row):
    assert digest(row['path'])==row['sha256']
    with np.load(row['path'],allow_pickle=False) as q:
        obs={'observation/image':q['image'],'observation/wrist_image':q['wrist_image'],
             'observation/state':q['state'],'prompt':str(q['prompt'].item())}
        noise=q['policy_noise']
    value=np.asarray(policy.infer(obs,noise=noise)['actions'])
    assert value.shape==(10,7) and np.isfinite(value).all()
    return value.copy()

baseline={suite:action(row) for suite,row in samples.items()}
checks=[];action_checks=[]
for suite,row in samples.items():
    captures={};expected={};pending=set();hooks=[]
    def pre(kind,module,args,kwargs):
        if kind in captures:return
        index=getattr(getattr(module,'self_attn',None),'layer_idx',0)
        # Native SigLIP can pass its mask positionally; normalize by the actual
        # signature rather than guessing argument ordering or dropping a mask.
        bound=dict(inspect.signature(module.forward).bind(*args,**kwargs).arguments)
        hidden=bound.pop('hidden_states')
        for key,parameter in inspect.signature(module.forward).parameters.items():
            if parameter.kind==inspect.Parameter.VAR_KEYWORD and key in bound:
                extra=bound.pop(key);assert not (set(extra)&set(bound));bound.update(extra)
        captures[kind]=capture_call((hidden,),bound,index)
        pending.add(kind)
    def post(kind,module,args,kwargs,result):
        if kind not in pending:return
        pending.remove(kind)
        value=result if isinstance(result,torch.Tensor) else result[0]
        expected[kind]=value.detach().cpu().clone()
    try:
        for kind,(name,module) in modules.items():
            hooks.append(module.register_forward_pre_hook(lambda m,x,k,kind=kind:pre(kind,m,x,k),with_kwargs=True))
            hooks.append(module.register_forward_hook(lambda m,x,k,y,kind=kind:post(kind,m,x,k,y),with_kwargs=True))
        hooked=action(row)
    finally:
        for hook in hooks:hook.remove()
    assert set(captures)==set(expected)==set(modules)
    np.savez_compressed(a.output/(suite+'-action-check.npz'),baseline=baseline[suite],hooked=hooked)
    action_checks.append(dict(suite=suite,input_sha256=row['sha256'],whole_policy_bitwise=bool(np.array_equal(baseline[suite],hooked))))
    for kind,(name,module) in modules.items():
        call=captures[kind]
        # This is calibration data, not a model checkpoint. Keep it in raw backup.
        path=a.output/(suite+'-'+kind+'.snapshot')
        torch.save(dict(captured=call,expected=expected[kind]),path)
        restored=torch.load(path,map_location='cpu',weights_only=True)
        one=replay_call(module,restored['captured']);one=one if isinstance(one,torch.Tensor) else one[0]
        two=replay_call(module,restored['captured']);two=two if isinstance(two,torch.Tensor) else two[0]
        one,two=one.detach().cpu(),two.detach().cpu()
        checks.append(dict(suite=suite,component=kind,module=name,shape=list(one.shape),
                           original_output_bitwise=bool(torch.equal(one,restored['expected'])),
                           repeated_replay_bitwise=bool(torch.equal(one,two)),
                           max_abs=float((one.float()-restored['expected'].float()).abs().max()),
                           snapshot_sha256=digest(path)))
    write_json(a.output/'progress.json',dict(action_checks=action_checks,block_checks=checks))
assert versions=={name:p._version for name,p in policy._model.named_parameters()}
passed=all(x['whole_policy_bitwise'] for x in action_checks) and all(x['original_output_bitwise'] and x['repeated_replay_bitwise'] for x in checks)
result=dict(gate='PASS_ACTUAL_PI05_NATIVE_BLOCK_REPLAY_CALIBRATION' if passed else 'FAIL_ACTUAL_PI05_NATIVE_BLOCK_REPLAY_CALIBRATION',
            backend=BACKEND_ID,calibration_only=True,reset_index=0,action_checks=action_checks,block_checks=checks,
            parameter_versions_unchanged=True,parameters_frozen=True,awq_algorithm_run=False,quantizer_admitted=False,
            formal_episodes=0,wall_seconds_including_load=time.monotonic()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),finished_at=now())
write_json(a.output/'result.json',result);print(json.dumps(result))
if not passed:raise RuntimeError('Native calibration block replay gate failed; no quantizer admission')
