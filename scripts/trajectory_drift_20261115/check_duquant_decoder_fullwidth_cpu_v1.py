"""CPU-only upstream decoder interface diagnostic; no model or method admission."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,hashlib,inspect,sys,traceback,types
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();source=a.root/'sources/DuQuant-archive-v1';sys.path[:0]=[str(source),str(Path(__file__).parent)]
import torch,transformers
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer,LlamaRotaryEmbedding
torch.set_num_threads(2);torch.manual_seed(7)
actual=json.loads((a.root/'models/openvla_oft/config.json').read_text());text=actual['text_config'];cfg=LlamaConfig(**text);cfg._attn_implementation='eager'
rows=[]
try:
 from duquant_llama_compat_v1 import load_decoder_class
 QuantLlamaDecoderLayer=load_decoder_class(source)
 for dtype in [torch.bfloat16]:
  layer=LlamaDecoderLayer(cfg,layer_idx=0).to(dtype=dtype).eval().requires_grad_(False);args=types.SimpleNamespace()
  for n in ['gate','down','up','k','v','q','o']:
   setattr(args,n+'_weight_quant_params',dict(n_bits=16,quant_method='none',dynamic_method='per_channel'));setattr(args,n+'_act_quant_params',dict(n_bits=16,quant_method='none',dynamic_method='per_token'))
  for n in ['q','k','p','v']:setattr(args,n+'_quant_params',dict(n_bits=16,quant_method='none',dynamic_method='per_token'))
  x=torch.randn(1,4,cfg.hidden_size,dtype=dtype);pos=torch.arange(4).unsqueeze(0);mask=torch.triu(torch.full((1,1,4,4),torch.finfo(dtype).min,dtype=dtype),diagonal=1)
  with torch.no_grad():baseline=layer(x,attention_mask=mask,position_ids=pos,use_cache=False)[0]
  row=dict(dtype=str(dtype),hf_reference_finite=bool(torch.isfinite(baseline).all()))
  try:
   wrapped=QuantLlamaDecoderLayer(cfg,layer,args).eval()
   with torch.no_grad():result=wrapped(x,attention_mask=mask,position_ids=pos,use_cache=False)[0]
   relative=float(((result.float()-baseline.float())**2).mean()/baseline.float().square().mean().clamp_min(1e-8));row.update(upstream_disabled_forward_pass=True,bitwise_equal=bool(torch.equal(result,baseline)),max_abs=float((result-baseline).abs().max()),relative_mse=relative,preregistered_relative_mse_max=1e-3);assert relative<=1e-3
   try:wrapped(x,position_ids=pos,use_cache=True)
   except ValueError:row['rejects_unadmitted_cache']=True
   else:raise AssertionError('Unadmitted cache accepted')
  except Exception as e:row.update(upstream_disabled_forward_pass=False,error=repr(e),traceback=traceback.format_exc(limit=4))
  rows.append(row)
except Exception as e:rows.append(dict(import_error=repr(e),traceback=traceback.format_exc(limit=4)))
out=dict(gate='CPU_FULLWIDTH_DECODER_COMPAT_ONLY_NOT_QUANTIZED_OR_MODEL_ADMISSION',transformers_version=transformers.__version__,actual_hidden_resolved=LlamaConfig(**text).hidden_size,actual_intermediate_resolved=LlamaConfig(**text).intermediate_size,fixture_hidden=cfg.hidden_size,fixture_intermediate=cfg.intermediate_size,synthetic_weights=True,full_7b_model_loaded=False,gpu_started=False,rotary_forward_signature=str(inspect.signature(LlamaRotaryEmbedding.forward)),tests=rows,source_sha256=hashlib.sha256((source/'models/int_llama_layer.py').read_bytes()).hexdigest())
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
