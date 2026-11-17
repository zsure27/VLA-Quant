"""Capture one native block from pinned calibration queries, never all layers.

This is preparation infrastructure, not full-policy AWQ admission. No labels or
development observations are used to choose calibration inputs.
"""
import inspect
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from qvla.infra.common import digest
from qvla.pi05_torch_backend_v2 import BACKEND_ID
from qvla.pi05_block_replay_v2 import capture_call
from qvla.pi05_awq_replay_batch_v1 import native_scale_groups
from qvla.pi05_awq_block_calibration_v1 import unique_tensor_bytes


def collect(policy,block,rows,max_cpu_bytes=2*1024**3):
    if len(rows)!=32 or len({r['sha256'] for r in rows})!=32 or len({r['suite'] for r in rows})!=4:
        raise ValueError('The fixed 32-query/four-suite calibration contract is required')
    if block.training or any(p.requires_grad for p in policy._model.parameters()):
        raise ValueError('The original frozen eval teacher is required')
    groups=native_scale_groups(block)
    linears={n:m for n,m in block.named_modules() if isinstance(m,nn.Linear) and not n.endswith('.dense')}
    # q/k/v and gate/up consume exactly the same activation; retain one copy.
    alias={n:n.replace('.k_proj','.q_proj').replace('.v_proj','.q_proj').replace('.up_proj','.gate_proj') for n in linears}
    features={n:[] for n in set(alias.values())};calls={g['score']:[] for g in groups};hooks=[];last_input={}
    versions={n:p._version for n,p in policy._model.named_parameters()}
    def check_budget():
        if unique_tensor_bytes([features,calls])>max_cpu_bytes:
            raise ValueError('Fixed calibration exceeds current-block memory budget')
    def score(name,module,args,kwargs):
        bound=dict(inspect.signature(module.forward).bind(*args,**kwargs).arguments)
        key=next((k for k in ['hidden_states','input','x'] if k in bound),None)
        if key is None:raise ValueError('Unknown native block input signature')
        hidden=bound.pop(key)
        for name_,parameter in inspect.signature(module.forward).parameters.items():
            if parameter.kind==inspect.Parameter.VAR_KEYWORD and name_ in bound:
                extra=bound.pop(name_)
                if set(extra)&set(bound):raise ValueError('Ambiguous native keyword arguments')
                bound.update(extra)
        calls[name].append(capture_call((hidden,),bound,getattr(module,'layer_idx',0)))
        check_budget()
    def feature(name,module,args):
        if len(args)!=1 or not isinstance(args[0],torch.Tensor):raise ValueError('Unknown linear input call')
        last_input[name]=args[0]
        features[name].append(args[0].detach().reshape(-1,args[0].shape[-1]).cpu().clone());check_budget()
    def check_alias(name,module,args):
        source=alias[name]
        if source not in last_input or len(args)!=1 or not torch.equal(last_input[source],args[0]):
            raise ValueError('Native shared projection inputs differ')
    try:
        for name in calls:
            hooks.append(block.get_submodule(name).register_forward_pre_hook(lambda m,x,k,name=name:score(name,m,x,k),with_kwargs=True))
        for name in features:
            hooks.append(block.get_submodule(name).register_forward_pre_hook(lambda m,x,name=name:feature(name,m,x)))
        for name,source in alias.items():
            if name!=source:hooks.append(block.get_submodule(name).register_forward_pre_hook(lambda m,x,name=name:check_alias(name,m,x)))
        for row in rows:
            path=Path(row['path'])
            if digest(path)!=row['sha256'] or row['role']!='calibration' or row['reset_index']!=0:
                raise ValueError('Calibration source identity or role changed')
            parent=json.loads((path.parent/'result.json').read_text())
            if not parent.get('gate','').startswith('PASS_PI05_TORCH_'):raise ValueError('Calibration episode was not admitted')
            if (parent['role'],parent['task_id'],parent['reset_index'],parent['condition'],parent['backend'],parent['suite'])!=('calibration',0,0,'bf16',BACKEND_ID,row['suite']):
                raise ValueError('Calibration provenance differs from fixed task0/reset0 teacher')
            with np.load(path,allow_pickle=False) as q:
                obs={'observation/image':q['image'],'observation/wrist_image':q['wrist_image'],'observation/state':q['state'],'prompt':str(q['prompt'].item())}
                expected=q['raw_policy_chunk'].copy();noise=q['policy_noise'].copy()
            output=np.asarray(policy.infer(obs,noise=noise)['actions'])
            if output.shape!=(10,7) or not np.isfinite(output).all() or not np.array_equal(output,expected):
                raise ValueError('Hooked teacher actions differ from the fixed BF16 calibration query')
    finally:
        for hook in hooks:hook.remove()
    if versions!={n:p._version for n,p in policy._model.named_parameters()}:
        raise ValueError('Calibration capture changed teacher parameters')
    joined={n:torch.cat(xs) for n,xs in features.items()}
    features={n:joined[source] for n,source in alias.items()}
    if unique_tensor_bytes([features,calls])>max_cpu_bytes:raise ValueError('Concatenated current-block budget exceeded')
    return calls,features
