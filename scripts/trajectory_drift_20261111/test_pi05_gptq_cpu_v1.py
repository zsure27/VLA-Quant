"""Small original-GPTQ CPU computation; CUDA synchronization is timing-only here."""
import argparse
import json
import math
from pathlib import Path
import sys
import types
import torch
from qvla.pi05_packed_weight_v1 import pack,unpack

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
sys.path.insert(0,str(a.root/'sources/gptq-archive-v1'));import gptq
torch.manual_seed(7);module=torch.nn.Linear(33,7,bias=False,dtype=torch.bfloat16)
module.requires_grad_(False);engine=gptq.GPTQ(module);engine.quantizer=gptq.Quantizer()
engine.quantizer.configure(4,perchannel=True,sym=False,mse=False)
scales=[];zeros=[];original=engine.quantizer.find_params
def record(self,value,weight=False):
    original(value,weight=weight)
    if value.shape[-1]<=16:scales.append(self.scale.clone());zeros.append(self.zero.clone())
engine.quantizer.find_params=types.MethodType(record,engine.quantizer)
engine.add_batch(torch.randn(1,64,33),None)
sync=torch.cuda.synchronize;torch.cuda.synchronize=lambda *args,**kwargs:None
try:engine.fasterquant(blocksize=16,percdamp=.01,groupsize=16,actorder=False,static_groups=False)
finally:torch.cuda.synchronize=sync
assert len(scales)==math.ceil(33/16)
entry=pack(module.weight,scales,zeros,16)
assert torch.equal(module.weight,unpack(entry,'cpu',torch.bfloat16))
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(dict(gate='PASS_ORIGINAL_GPTQ_SMALL_CPU_PACK',shape=[7,33],groups=3,
    ragged_last_group=True,odd_nibble_count=True,gpu_model_admitted=False),indent=2)+'\n')
print('PASS_ORIGINAL_GPTQ_SMALL_CPU_PACK')
