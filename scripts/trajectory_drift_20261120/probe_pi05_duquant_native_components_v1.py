"""Original DuQuant on three real pi05 projections: component gate, not rollout.

Uses one fixed calibration query per suite, at most first 64 real input tokens,
original 1024 rotation steps and one zigzag permutation; no trained adapter.
This does NOT create an admitted 414-layer quantized policy.
"""
import argparse,gc,hashlib,importlib,json,os,random,shutil,sys,time
from pathlib import Path
import numpy as np
import torch


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(path,body):path.write_text(json.dumps(body,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',type=Path,required=True);p.add_argument('--suite',required=True)
    a=p.parse_args();started=time.monotonic();out=a.stage/('components-'+a.suite)
    out.mkdir(exist_ok=False);assert torch.cuda.is_available()
    torch.set_num_threads(4);random.seed(7);np.random.seed(7);torch.manual_seed(7)
    src=a.root/'sources/DuQuant-archive-v1';expected=json.loads((a.root/'preparations/admission-first-20261119-v1/readiness/readiness.json').read_text())
    index={row['path']:row['sha256'] for source in expected['source_directories'] for row in source.get('files',[])}
    for name in ['quantize/int_linear.py','quantize/quantizer.py','quantize/const.py','utils.py']:
        assert sha(src/name)==index[str((src/name).relative_to(a.root))]
    assets=a.root/'preparations/pi05-duquant-20261112-v3';proof=json.loads((assets/'result.json').read_text())
    assert proof['gate']=='PASS_ORIGINAL_DUQUANT_DIVISIBLE_BLOCKS_ROTATION_RELOAD_CPU'
    assert sha(assets/'Rot.pkl')==proof['rotation_asset_sha256']
    shutil.copyfile(assets/'Rot.pkl',out/'Rot.pkl');os.chdir(out)
    sys.path.insert(0,str(src))
    original=importlib.import_module('quantize.int_linear');quantizer=importlib.import_module('quantize.quantizer')
    assert Path(original.__file__).resolve()==src/'quantize/int_linear.py'
    assert Path(quantizer.__file__).resolve()==src/'quantize/quantizer.py'
    cache_dir=a.root/'rounds/20261111-084-pi05-smoothquant-a8-v2/capture'
    cap_receipt=json.loads((cache_dir/'result.json').read_text())
    assert sha(cache_dir/'statistics.pt')==cap_receipt['profile_sha256']
    cache=torch.load(cache_dir/'statistics.pt',map_location='cpu',weights_only=True)
    rows=cache['samples'];assert len(rows)==32
    row=next(x for x in rows if x['suite']==a.suite);assert sha(row['path'])==row['sha256']
    # Parent receipts bind these samples to calibration reset0. Never eval labels.
    from qvla.pi05_torch_backend_v2 import load_policy,target_names
    policy=load_policy(a.root);names=target_names(policy._model);assert len(names)==414
    selected=[next(n for n in names if '.vision_tower.' in n and n.endswith('.mlp.fc2')),
              next(n for n in names if '.language_model.' in n and n.endswith('.self_attn.q_proj')),
              next(n for n in names if '.gemma_expert.' in n and n.endswith('.self_attn.q_proj'))]
    inputs={};handles=[];weights={}
    for name in selected:
        layer=policy._model.get_submodule(name)
        weights[name]=(layer.weight.detach().float().cpu().clone(),None if layer.bias is None else layer.bias.detach().float().cpu().clone())
        def capture(module,args,key=name):
            if key not in inputs:
                x=args[0].detach();assert torch.isfinite(x).all()
                inputs[key]=dict(value=x.reshape(-1,x.shape[-1])[:64].float().cpu().clone(),native_shape=list(x.shape))
        handles.append(layer.register_forward_pre_hook(capture))
    with np.load(row['path'],allow_pickle=False) as q:
        obs={'observation/image':q['image'],'observation/wrist_image':q['wrist_image'],'observation/state':q['state'],'prompt':str(q['prompt'].item())}
        noise=q['policy_noise'].copy();teacher=q['raw_policy_chunk'].copy()
    actual=np.asarray(policy.infer(obs,noise=noise)['actions']).copy()
    assert np.array_equal(actual,teacher),'Cloned native teacher first calibration query differs'
    for handle in handles:handle.remove()
    assert set(inputs)==set(selected)
    del handles,handle,layer,policy;gc.collect();torch.cuda.empty_cache()
    previous_tf32=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False
    results=[]
    try:
        for i,name in enumerate(selected):
            weight,bias=weights[name];x=inputs[name]['value'].cuda();width=weight.shape[1]
            block=128 if width%128==0 else 16;assert width%block==0
            base=torch.nn.Linear(width,weight.shape[0],bias=bias is not None,device='cuda',dtype=torch.float32)
            with torch.no_grad():
                base.weight.copy_(weight.cuda())
                if bias is not None:base.bias.copy_(bias.cuda())
            base.requires_grad_(False);base.eval();before=base.weight.detach().clone()
            kwargs=dict(n_bits=4,dynamic_method='per_channel',quant_method='duquant',group_size=block,
                        block_size=block,max_rotation_step=1024,permutation_times=1,lwc=False)
            akw=dict(n_bits=8,dynamic_method='per_token',quant_method='duquant',block_size=block,
                     max_rotation_step=1024,permutation_times=1)
            module=original.QuantLinear(base,weight_quant_params=kwargs,act_quant_params=akw,rotate=True).cuda().eval()
            module.requires_grad_(False)
            with torch.no_grad():
                expected_y=base(x)
                # Original rotations are learned from the real input, with all
                # quantization disabled; shared weight transform is copied.
                xr=module.act_quantizer.init_duquant(x.clone())
                module.weight_quantizer.copy_duquant_params(module.act_quantizer)
                module.init_duquant_params=torch.tensor(1)
                wr=module.weight_quantizer.init_duquant(module.weight.clone())
                equivalent=torch.nn.functional.linear(xr,wr,module.bias)
                error=(equivalent-expected_y).float()
                normalized_rmse=float(error.square().mean().sqrt()/expected_y.float().square().mean().sqrt().clamp_min(1e-8))
                normalized_max=float(error.abs().max()/expected_y.float().abs().max().clamp_min(1e-8))
                assert normalized_rmse<=5e-4 and normalized_max<=5e-3,(name,normalized_rmse,normalized_max)
                assert torch.equal(base.weight,before),'Base weight changed'
                module.set_quant_state(True,True)
                values={}
                for bits in [8,4]:
                    module.act_quantizer.change_n_bits(bits);y=module(x.clone())
                    assert y.shape==expected_y.shape and torch.isfinite(y).all()
                    values[str(bits)]=y.detach().cpu().clone()
                module.act_quantizer.change_n_bits(8)
                reference=module(x.clone()).detach().cpu()
                module.weight_quantizer.register_duquant_params();module.act_quantizer.register_duquant_params()
                saved=out/f'component-{i:02d}.pt';torch.save(module.state_dict(),saved)
                restored=original.QuantLinear(base,weight_quant_params=kwargs,act_quant_params=akw,rotate=True).cuda().eval()
                restored.copy_quantizers_duquant_params(module)
                restored.weight_quantizer.register_duquant_params();restored.act_quantizer.register_duquant_params()
                restored.load_state_dict(torch.load(saved,map_location='cuda',weights_only=True));restored.set_quant_state(True,True)
                assert torch.equal(restored(x.clone()).detach().cpu(),reference),'Original DuQuant save/reload mismatch'
                assert torch.equal(base.weight,before)
            results.append(dict(name=name,input_shape=inputs[name]['native_shape'],probe_tokens=x.shape[0],
                weight_shape=list(weight.shape),rotation_block=block,rotation_steps=1024,permutation_times=1,
                fp32_transform_normalized_rmse=normalized_rmse,fp32_transform_normalized_max=normalized_max,
                a8_a4_finite=True,base_frozen=True,reload_bitwise=True,artifact_sha256=sha(saved)))
            write(out/'progress.json',dict(completed_components=results,formal_episodes=0,full_model_admitted=False))
            del module,restored,base,x,xr,wr,before,expected_y,equivalent,error;gc.collect();torch.cuda.empty_cache()
    finally:torch.backends.cuda.matmul.allow_tf32=previous_tf32
    result=dict(gate='PASS_PI05_ORIGINAL_DUQUANT_THREE_NATIVE_PROJECTION_GPU_COMPONENTS',suite=a.suite,
                teacher_calibration_query_bitwise=True,calibration_input_sha256=row['sha256'],
                components=results,formal_episodes=0,full_model_admitted=False,
                wall_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    write(out/'result.json',result)
    print(json.dumps({k:result[k] for k in ['gate','suite','formal_episodes','full_model_admitted','wall_seconds']}))


if __name__=='__main__':main()
