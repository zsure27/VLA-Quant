"""Lossless nibble packing; reused arithmetic, independent pi0.5 identity."""
import hashlib
import torch

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

