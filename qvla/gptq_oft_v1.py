"""Lossless packing of original GPTQ's dequantized weights; no INT4 kernel claim."""
from pathlib import Path
import hashlib
import torch
from qvla.baseline_contract import check_scope, CONTRACT_ID

def weight_sha(weight):
    return hashlib.sha256(weight.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

def unpack(entry, device, dtype):
    rows, cols = entry['matrix_shape']
    packed = entry['packed'].to(device)
    values = torch.stack((packed & 15, packed >> 4), dim=1).flatten()[:rows*cols].reshape(rows, cols).float()
    index = torch.arange(cols, device=device)//entry['group_size']
    scale = entry['scales'].to(device)[:, index]
    zero = entry['zeros'].to(device)[:, index]
    return (scale*(values-zero)).to(dtype).reshape(entry['shape'])

def pack(weight, scales, zeros, group_size):
    shape=list(weight.shape);w=weight.detach().flatten(1).float();rows,cols=w.shape
    scales=torch.stack([x.reshape(rows) for x in scales],dim=1).to(w.device)
    zeros=torch.stack([x.reshape(rows) for x in zeros],dim=1).to(w.device)
    assert scales.shape[1]==(cols+group_size-1)//group_size
    idx=torch.arange(cols,device=w.device)//group_size
    q=(w/scales[:,idx]+zeros[:,idx]).round();assert q.min()>=0 and q.max()<=15
    q=q.to(torch.uint8).flatten()
    if q.numel()%2:q=torch.cat([q,torch.zeros(1,dtype=torch.uint8,device=q.device)])
    result=dict(shape=shape,matrix_shape=[rows,cols],group_size=group_size,
                packed=(q[::2] | (q[1::2]<<4)).cpu(),scales=scales.cpu(),zeros=zeros.cpu(),weight_sha256=weight_sha(weight))
    assert torch.equal(unpack(result,weight.device,weight.dtype),weight),'Packing changed original GPTQ output'
    return result

def load_profiles(paths, method):
    assert method=='gptq' and len(paths)==1
    payload=torch.load(Path(paths[0]),map_location='cpu',weights_only=True)
    assert payload['format']=='OFT_ORIGINAL_GPTQ_PACK_V1' and payload['contract_id']==CONTRACT_ID
    assert payload['gate']=='PASS_FULL_422_GPTQ_PACK_AND_RELOAD'
    check_scope(payload['entries'])
    return payload['entries'],payload

@torch.no_grad()
def apply_quantization(model, entries, metadata, method, official, weight_bits, activation_bits):
    assert method=='gptq' and weight_bits==4 and activation_bits==16
    check_scope(entries);modules=dict(model.named_modules())
    for name,entry in entries.items():
        module=modules[name];assert list(module.weight.shape)==entry['shape']
        weight=unpack(entry,module.weight.device,module.weight.dtype)
        assert torch.isfinite(weight).all() and weight_sha(weight)==entry['weight_sha256']
        module.weight.copy_(weight)
    return []

@torch.no_grad()
def validate_calibration_actions(model, head, proprio, processor, cfg, metadata, directory):
    from qvla.action_jacobian_batch import load_sample, prepare_inputs
    import json
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
    assert manifest['gate']=='PASS_DISTINCT_RESET0_CALIBRATION_CAPTURE'
    expected=metadata['expected_calibration_actions'];assert len(expected)==len(manifest['samples'])==4
    old=cfg.unnorm_key
    try:
        for row,action in zip(manifest['samples'],expected):
            path=directory/row['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==metadata['sample_sha256'][path.name]
            cfg.unnorm_key=row['suite']+'_no_noops'
            inputs,state=prepare_inputs(load_sample(path),cfg,model,processor,next(model.parameters()).device)
            value=model.predict_action(**inputs,unnorm_key=cfg.unnorm_key,do_sample=False,
                proprio=state,proprio_projector=proprio,action_head=head,noisy_action_projector=None,use_film=False)
            actual=torch.as_tensor(value[0] if isinstance(value,tuple) else value).cpu()
            assert torch.equal(actual,action),'Cross-process packed artifact action replay changed'
    finally:cfg.unnorm_key=old
    print('PASS_GPTQ_ACTUAL_MODEL_CALIBRATION_ACTION_REPLAY',flush=True)
