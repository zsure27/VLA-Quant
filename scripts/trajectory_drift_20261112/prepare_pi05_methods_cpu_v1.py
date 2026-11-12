"""Memory-bounded preparation; upstream component checks are NOT model admission.

Reads safetensors metadata only. No policy/model allocation, CUDA or rollout.
New outputs only; all algorithm source files remain unchanged.
"""
import argparse,ast,collections,datetime,hashlib,importlib.util,json,os,pathlib,re,struct,subprocess,sys,traceback

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def load_file(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def component(root,out,method):
 import torch
 torch.set_num_threads(1);torch.manual_seed(7)
 assert not torch.cuda.is_available(),'CPU preparation requires GPU absent/hidden'
 if method=='awq':
  src=root/'sources/quant-source-sparse-v2/llm-awq/awq/quantize/quantizer.py'
  normalized=hashlib.sha256(src.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
  assert normalized=='7929ee1f97addde95ce1b9c1d55f027b3b4d13fb1b9e6c850ab8a01736914e6c'
  tree=ast.parse(src.read_text());node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='pseudo_quantize_tensor')
  namespace={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(src),'exec'),namespace)
  w=torch.randn(9,128);before=w.clone();fn=namespace[node.name]
  quant=fn(w.clone(),n_bit=4,zero_point=True,q_group_size=128,inplace=False)
  assert torch.isfinite(quant).all() and not torch.equal(quant,w) and torch.equal(w,before)
  file=out/'awq-small-tensor.pt';torch.save(quant,file);assert torch.equal(torch.load(file,weights_only=True),quant)
  try:fn(torch.randn(3,129),n_bit=4,zero_point=True,q_group_size=128)
  except AssertionError:pass
  else:raise AssertionError('Invalid group width was accepted')
  result=dict(gate='PASS_PINNED_AWQ_PRIMITIVE_CPU',normalized_source_sha256=normalized,source_sha256=digest(src),unchanged_ast_function=True,shape=[9,128],invalid_width_rejected=True,activation_search_tested=False,full_model_adapter_complete=False)
 elif method=='omniquant':
  src=root/'sources/OmniQuant-archive-v1/quantize/quantizer.py';m=load_file('_cpu_omni_quantizer',src)
  w=torch.randn(9,32);before=w.clone();x=torch.randn(16,32);teacher=torch.nn.functional.linear(x,w).detach()
  def make():return m.UniformAffineQuantizer(n_bits=4,symmetric=False,dynamic_method='per_channel',group_size=16,shape=list(w.shape),lwc=True)
  q=make();optim=torch.optim.Adam(q.parameters(),lr=.01)
  y=q(w);loss=torch.nn.functional.mse_loss(torch.nn.functional.linear(x,y),teacher);loss.backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in q.parameters())
  assert any(p.grad.abs().max()>0 for p in q.parameters())
  optim.step();assert torch.equal(w,before) and w.grad is None
  file=out/'omni-small-state.pt';torch.save(q.state_dict(),file);q2=make();q2.load_state_dict(torch.load(file,weights_only=True))
  assert torch.equal(q(w),q2(w)) and torch.isfinite(q(w)).all()
  result=dict(gate='PASS_ORIGINAL_OMNI_LWC_GRADIENT_FREEZE_RELOAD_CPU',source_sha256=digest(src),shape=[9,32],weight_frozen=True,clipping_gradient_nonzero=True,full_block_reconstruction_tested=False,full_model_adapter_complete=False)
 elif method=='duquant':
  src=root/'sources/DuQuant-archive-v1';os.chdir(src);sys.path.insert(0,str(src))
  from quantize.quantizer import UniformAffineQuantizer
  # Original dual rotation needs its pinned Rot.pkl. Do not substitute a random
  # rotation/Hadamard and mislabel it as the complete DuQuant algorithm.
  if not (src/'Rot.pkl').exists():
   result=dict(gate='BLOCKED_ORIGINAL_DUQUANT_ROTATION_ASSET',source_sha256=digest(src/'quantize/quantizer.py'),missing='Rot.pkl',full_model_adapter_complete=False,model_inapplicability_proven=False)
  else:
   q=UniformAffineQuantizer(n_bits=4,dynamic_method='per_token',quant_method='duquant',rotate=True,block_size=128,max_rotation_step=4,permutation_times=1)
   x=torch.randn(16,128);xr=q.init_duquant(x.clone());basis=q.init_duquant(torch.eye(128));w=torch.randn(9,128)
   assert torch.allclose(torch.nn.functional.linear(xr,w@basis),torch.nn.functional.linear(x,w),atol=1e-4,rtol=1e-4)
   q.register_duquant_params();state=out/'duquant-small-state.pt';torch.save(q.state_dict(),state)
   q2=UniformAffineQuantizer(n_bits=4,dynamic_method='per_token',quant_method='duquant',rotate=True,block_size=128,max_rotation_step=4,permutation_times=1)
   q2.copy_duquant_params(q);q2.register_duquant_params();q2.load_state_dict(torch.load(state,weights_only=True))
   assert torch.equal(q(x.clone()),q2(x.clone()))
   values={}
   for bits in [4,8]:
    q.change_n_bits(bits);y=q(x.clone());assert torch.isfinite(y).all();values[str(bits)]=float((y-xr).square().mean())
   result=dict(gate='PASS_ORIGINAL_DUQUANT_ROTATION_AND_RELOAD_CPU',source_sha256=digest(src/'quantize/quantizer.py'),rotation_asset_sha256=digest(src/'Rot.pkl'),toy_rotation_steps=4,full_model_adapter_complete=False,activation_toy_mse=values)
 else:raise ValueError(method)
 result.update(method=method,gpu_model_admitted=False,real_trajectory_completed=False)
 save(out/(method+'-cpu.json'),result);print(json.dumps(result))

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--component',choices=['awq','omniquant','duquant']);a=p.parse_args()
 root,out=a.root.resolve(),a.output.resolve()
 if a.component:return component(root,out,a.component)
 out.mkdir(parents=True,exist_ok=False);os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1'
 file=root/'models/pi05_libero_pytorch_bridge_v2/model.safetensors'
 with file.open('rb') as f:
  size=struct.unpack('<Q',f.read(8))[0];assert size<32*1024**2;header=json.loads(f.read(size))
 # AST-extract the existing frozen target selector's literal patterns.
 backend=root/'code-084-pi05-smoothquant-v2/qvla/pi05_torch_backend_v2.py'
 tree=ast.parse(backend.read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='target_names')
 patterns=ast.literal_eval(next(x.value for x in fn.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='patterns' for t in x.targets)))
 targets=[]
 for key,value in header.items():
  if key.endswith('.weight') and any(re.fullmatch(p,key[:-7]) for p in patterns):
   shape=value['shape'];assert len(shape)==2
   targets.append(dict(name=key[:-7],shape=shape,dtype=value['dtype'],group128_divisible=shape[1]%128==0,hessian_fp32_bytes=4*shape[1]**2))
 assert len(targets)==414,len(targets)
 save(out/'target-contract.json',dict(gate='PASS_PI05_FROZEN_HEADER_414_TARGET_SCOPE',backend_source_sha256=digest(backend),safetensors_header_sha256=hashlib.sha256(json.dumps(header,sort_keys=True).encode()).hexdigest(),target_parameters=sum(x['shape'][0]*x['shape'][1] for x in targets),targets=targets,excluded='embedding, conv, norm, conditioning, action input/output, unused vocabulary head',policy_loaded=False))
 results={}
 for method in ['awq','omniquant','duquant']:
  with (out/(method+'-cpu.log')).open('w') as log:
   proc=subprocess.run([sys.executable,str(pathlib.Path(__file__).resolve()),'--root',str(root),'--output',str(out),'--component',method],stdout=log,stderr=subprocess.STDOUT,timeout=120,env=os.environ)
  result=out/(method+'-cpu.json')
  results[method]=json.loads(result.read_text()) if proc.returncode==0 and result.exists() else dict(gate='CPU_COMPONENT_FAILED',exit_code=proc.returncode,log=str(out/(method+'-cpu.log')),gpu_model_admitted=False)
 # Explicit, non-executable preparation contracts; no model compatibility claims.
 methods=[('gptq',16),('awq',16),('omniquant',16),('actquant',16),('qvla',16),('smoothquant',8),('duquant',8),('quantvla',8),('smoothquant',4),('duquant',4),('quantvla',4),('holoq_vla',4)]
 plans=[]
 for method,abits in methods:
  pending=['actual full-model target coverage','same-observation/noise finite action check','frozen-base and save/reload check','four-suite task0/reset5 paired pilot']
  if method=='awq':pending.insert(0,'original AWQ block forward/search binding for SigLIP, Gemma and adaptive expert; primitive-only check is insufficient')
  if method=='omniquant':pending.insert(0,'native flow/block reconstruction and bounded teacher cache; LWC primitive alone is insufficient')
  if method=='duquant':pending.insert(0,'original rotation asset and model-wide paired activation/weight rotation before quantization')
  if method in ['actquant','qvla','quantvla','holoq_vla']:pending.insert(0,'pinned applicable original implementation and pi0.5 architecture adapter; do not relabel another method')
  plans.append(dict(id=f'{method}_w4a{abits}',weight_bits=4,activation_bits=abits,backend='openpi-pytorch-original-independent-h10-v2',target_contract='target-contract.json',calibration_resets=[0],evaluation_resets=[5,15,25,35,45],suites=['libero_spatial','libero_object','libero_goal','libero_10'],pilot_tasks=[0],finetuning='none',gpu_execution_authorized=False,executable=False,pending=pending))
 save(out/'gpu-preparation-contracts.json',plans)
 save(out/'result.json',dict(gate='PASS_PI05_CPU_SCOPE_AND_METHOD_READINESS_INVENTORY',cpu_only=True,policy_loaded=False,gpu_authorized=False,targets=414,component_results=results,configurations=12,all_methods_gpu_ready=False,observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_script_sha256=digest(pathlib.Path(__file__).resolve())))
 print(json.dumps(dict(output=str(out),targets=414,components={k:v['gate'] for k,v in results.items()},gpu_model_admitted=False)))
if __name__=='__main__':main()
