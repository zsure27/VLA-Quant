"""Compile actual CPU evidence into a next-GPU handoff; never launch GPU."""
import argparse,ast,hashlib,importlib.util,json,os,pathlib,subprocess,sys,tarfile
import torch
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--layout-module',type=pathlib.Path,required=True);a=p.parse_args()
r,out=a.root.resolve(),a.output.resolve();out.mkdir(parents=True,exist_ok=False);os.environ['CUDA_VISIBLE_DEVICES']='';torch.set_num_threads(1);torch.manual_seed(7)
assert not torch.cuda.is_available()
spec=importlib.util.spec_from_file_location('pi05_layout',a.layout_module);layout=importlib.util.module_from_spec(spec);spec.loader.exec_module(layout)
parent=r/'preparations/pi05-methods-20261112-v1';targets=json.loads((parent/'target-contract.json').read_text())['targets'];rows=layout.layout(targets)
assert {x['awq_group_size'] for x in rows}=={16,128}
src=r/'sources/quant-source-sparse-v2/llm-awq/awq/quantize/quantizer.py';tree=ast.parse(src.read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='pseudo_quantize_tensor');ns={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(src),'exec'),ns)
spec=importlib.util.spec_from_file_location('_pi05_omni_layout',r/'sources/OmniQuant-archive-v1/quantize/quantizer.py');omni=importlib.util.module_from_spec(spec);spec.loader.exec_module(omni)
checks=[]
for width in [128,4304]:
 group=layout.divisible_group(width);w=torch.randn(9,width);original=w.clone()
 q=ns['pseudo_quantize_tensor'](w.clone(),n_bit=4,zero_point=True,q_group_size=group,inplace=False);assert q.shape==w.shape and torch.isfinite(q).all()
 def make():return omni.UniformAffineQuantizer(n_bits=4,symmetric=False,dynamic_method='per_channel',group_size=group,shape=list(w.shape),lwc=True)
 o=make();loss=(o(w)-w).square().mean();loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in o.parameters())
 file=out/f'omni-layout-{width}.pt';torch.save(o.state_dict(),file);o2=make();o2.load_state_dict(torch.load(file,weights_only=True));assert torch.equal(o(w),o2(w)) and torch.equal(w,original)
 checks.append(dict(input_width=width,divisible_group=group,awq_finite=True,omni_gradient_and_reload=True))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=dict(gate='PASS_PI05_414_LAYOUT_AND_REAL_WIDTH_CPU_HANDOFF',gpu_authorized=False,gpu_experiments_started=0,full_model_admitted=False,targets=414,group128_targets=387,group16_vision_fc2_targets=27,layout_policy_sha256=sha(a.layout_module),checks=checks,duquant=json.loads((r/'preparations/pi05-duquant-20261112-v3/result.json').read_text()),model_and_existing_profile_verification=json.loads((parent/'reopen-and-artifact-verification.json').read_text()),sources=[{k:x[k] for k in ['repository','commit','archive_bytes','archive_sha256','selected_source_bytes']} for x in json.loads((r/'preparations/optional-method-sources-20261112-v2/source-verification.json').read_text())],ready_for_next_gpu=['SQ W4A4: reuse verified 4MiB profile, real first-case guard then four-suite pilot','GPTQ W4A16: existing original layer-shard builder and runner; actual Hessian VRAM/full quantization/finite/reload still required'],full_model_adapter_work_remaining=['AWQ original block search for Gemma/adaRMS/SigLIP; no per-linear substitute','OmniQuant native block reconstruction; primitive LWC alone insufficient','DuQuant paired weight/activation rotations through real native execution','ActQuant C++ BF16 reference bridge and >50GB scratch budget; no Torch cross-backend subtraction','QVLA published OFT workflow lacks pi0.5 implementation; adaptation not admitted','QuantVLA ATM/OHB pi0.5 flow binding absent in GR00T-focused source','HoloQ-VLA faithful pi0.5 implementation unconfirmed; no placeholder algorithm'])
(out/'layout.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(gate=report['gate'],targets=414,width4304_group=16,gpu_experiments_started=0,full_model_admitted=False)))
