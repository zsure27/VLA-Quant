"""Header-only conservative memory bound for the original all-target AMF path."""
import argparse,ast,json,pathlib,struct,subprocess,hashlib,math
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--control',type=pathlib.Path,required=True);a=p.parse_args();r=a.root.resolve();out=a.control/'cpu-preparation/actquant-memory-v1';out.mkdir(parents=True,exist_ok=False)
src=r/'sources/ActQuant-original-b64791125070/tools/fisher-diag/compute_amf_lq.py';body=src.read_text();tree=ast.parse(body);funcs={x.name:ast.get_source_segment(body,x) for x in tree.body if isinstance(x,ast.FunctionDef)}
assert 'torch.bfloat16' in funcs['load_vla'] and 'target_ids' in funcs['freeze_except_targets'] and 'get_target_params(vla' in funcs['_gpu_worker']
rows=[];base=0;grads=0
for f in sorted((r/'models/openvla_oft').glob('*.safetensors')):
 with f.open('rb') as stream:size=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(size))
 for name,spec in header.items():
  if name=='__metadata__' or spec['dtype'] not in ['BF16','F16','F32','F64']:continue
  n=math.prod(spec['shape']);base+=n*2
  target=name.startswith('language_model.') and len(spec['shape'])==2 and 'embed_tokens' not in name and 'norm' not in name
  if target:grads+=n*2;rows.append(dict(name=name,shape=spec['shape'],bf16_gradient_bytes=n*2))
vram_mib=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True).strip());limit=vram_mib*1024**2;bound=base+grads
result=dict(gate='BLOCK_ORIGINAL_ACTQUANT_AMF_DEFAULT_VRAM_LOWER_BOUND' if bound>limit else 'AMF_LOWER_BOUND_ONLY_NOT_GPU_ADMISSION',base_bf16_weight_lower_bound_bytes=base,llm_bf16_gradient_lower_bound_bytes=grads,sum_lower_bound_bytes=bound,gpu_total_bytes=limit,exceeds_total_vram=bound>limit,llm_target_matrices=len(rows),original_source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),original_multigpu='Each worker calls get_target_params on full model; splits frames, not model/gradient shards',excluded_costs=['vision gradients','action/proprio heads','forward/backward activations','CUDA workspaces','allocator overhead'],qualification='Static bound for default published all-target BF16 path; not measured OOM, not theoretical method incompatibility. CPU-offload/streamed-gradient adaptation would require independent equivalence admission.',no_model_loaded=True,no_gpu_calibration_attempted=True,targets=rows)
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='targets'}))
