"""Apply original-GPTQ layer shards to the independent same-Torch cohort."""
from pathlib import Path
import torch
from qvla.infra.common import digest
from qvla.pi05_torch_backend_v2 import BACKEND_ID,target_names
from qvla.pi05_packed_weight_v1 import unpack,weight_sha


@torch.no_grad()
def apply(model,payload,profile_path):
    if (payload.get('format')!='PI05_ORIGINAL_GPTQ_PACK_V1' or payload.get('backend')!=BACKEND_ID
            or payload.get('gate')!='PASS_PI05_GPTQ_414_PACK_RELOAD'):
        raise ValueError('GPTQ artifact backend/admission mismatch')
    entries=payload['entries'];names=target_names(model)
    if len(entries)!=414 or {x['name'] for x in entries}!=set(names):raise ValueError('GPTQ target scope mismatch')
    modules=dict(model.named_modules());base=Path(profile_path).parent.resolve()
    for record in entries:
        file=(base/record['file']).resolve()
        if base not in file.parents or digest(file)!=record['sha256']:raise ValueError('GPTQ shard identity/path mismatch')
        saved=torch.load(file,map_location='cpu',weights_only=True);name=record['name'];module=modules[name]
        if saved['name']!=name or saved['entry']['shape']!=list(module.weight.shape):raise ValueError('GPTQ module mismatch')
        weight=unpack(saved['entry'],module.weight.device,module.weight.dtype)
        if weight_sha(weight)!=record['weight_sha256'] or not torch.isfinite(weight).all():raise ValueError('GPTQ weight roundtrip mismatch')
        module.weight.copy_(weight)
    return len(entries)
