from pathlib import Path
import os,sys,json,torch
r=Path('/root/autodl-tmp/trajectory-drift-20261107');code=r/'code-cpu-prep-20261113-v1';root=r/'control/20261113-061-cpu-quant-ready-v1';os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,str(code));from qvla.duquant_oft_core_v1 import load_original,FrozenDuQuantLinear,digest
assert not torch.cuda.is_available();torch.set_num_threads(1);torch.manual_seed(7);asset=root/'duquant-core-cpu-v1/rotations-weights-only.pt';Q=load_original(r/'sources/DuQuant-archive-v1',asset,digest(asset));checks=[]
for width,block in [(128,128),(4304,16),(588,4)]:
 state=torch.load(root/f'duquant-core-cpu-v1/fixture-width{width}-a8.pt',weights_only=True);profile=dict(width=width,block=block,calibration_role='calibration_reset0',source_sha256='99c2a0d76866e2289658fb85a2b7e2673c23517fb1d9c68642753bf5b9830b1a',rotation=state['act_quantizer.R'],permutation=state['act_quantizer.permutation_list']);layer=torch.nn.Linear(width,9).to(torch.bfloat16)
 with torch.no_grad():layer.weight.copy_(state['weight']);layer.bias.copy_(state['bias'])
 x=torch.randn(2,3,width).bfloat16();res={}
 for bits in [8,4]:
  m=FrozenDuQuantLinear(layer,Q,profile,activation_bits=bits);before=x.clone()
  with torch.autocast('cpu',dtype=torch.bfloat16):
   z=m(x);ref=layer(x);assert torch.equal(z,ref);m.enabled=True;out=m(x);assert torch.isfinite(out).all() and out.dtype==torch.bfloat16
  assert torch.equal(before,x);res[str(bits)]=dict(finite=True,output_dtype=str(out.dtype),disabled_bitwise=True)
 checks.append(dict(width=width,block=block,precisions=res))
result=dict(gate='PASS_BF16_AUTOCAST_DUQUANT_CPU_COMPONENT_ONLY',checks=checks,gpu_or_model_admitted=False,qualification='Small CPU BF16 autocast fixtures only; GPU full action/rotation equivalence still required');(root/'DUQUANT_BF16_CPU_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));scope=json.loads((root/'inventory/duquant-scope422.json').read_text());from collections import Counter;print('SCOPE_COUNTS',dict(Counter(x['kind'] for x in scope)),dict(Counter((x['flattened_input_width'],x['rotation_block']) for x in scope)))
