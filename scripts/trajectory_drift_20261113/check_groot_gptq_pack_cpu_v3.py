import os,sys,json,types,math,hashlib,argparse
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2'
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--code',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();r=args.root;source=r/'sources/gptq-upstream-v1';sys.path[:0]=[str(source),str(args.code)]
import torch
from gptq import GPTQ
from quant import Quantizer
from qvla.pi05_packed_weight_v1 import pack,unpack
assert not torch.cuda.is_available();torch.set_num_threads(2)
original_sync=torch.cuda.synchronize;torch.cuda.synchronize=lambda *a,**k:None
rows=[]
try:
 for width in [64,128,129,256]:
  torch.manual_seed(7);layer=torch.nn.Linear(width,4,bias=True);engine=GPTQ(layer);engine.quantizer=Quantizer();engine.quantizer.configure(4,perchannel=True,sym=False,mse=False);scales=[];zeros=[];original=engine.quantizer.find_params;calls=[0]
  def find(self,value,*a,**kw):
   original(value,*a,**kw);calls[0]+=1
   if calls[0]>1:
    assert value.shape[-1]<=128;scales.append(self.scale.detach().clone());zeros.append(self.zero.detach().clone())
  engine.quantizer.find_params=types.MethodType(find,engine.quantizer);engine.add_batch(torch.randn(1,128,width),None);engine.fasterquant(blocksize=128,percdamp=.01,groupsize=128,actorder=False,static_groups=False);assert len(scales)==math.ceil(width/128)
  assert torch.isfinite(layer.weight).all();entry=pack(layer.weight,scales,zeros,128);assert torch.equal(unpack(entry,'cpu',layer.weight.dtype),layer.weight)
  rows.append(dict(width=width,groups=len(scales),pack_bitwise_equal=True,finite=True))
finally:torch.cuda.synchronize=original_sync
out=dict(gate='PASS_GROOT_GPTQ_SCALE_CAPTURE_AND_PACK_CPU',model_loaded=False,gpu_used=False,tests=rows,full_model_gpu_admitted=False,source_sha256={f:hashlib.sha256((source/f).read_bytes()).hexdigest() for f in ['gptq.py','quant.py']})
p=args.output;p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
