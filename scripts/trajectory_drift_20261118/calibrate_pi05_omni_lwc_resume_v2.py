"""LWC-only native-unit adaptation; full-policy admission is separate from smoke."""
import argparse,gc,json,time
from pathlib import Path
import numpy as np
import torch
from qvla.infra.common import digest,write_json,now
from qvla.baseline_contract import checkpoint_identity
from qvla.pi05_torch_backend_v2 import load_policy,BACKEND_ID
from qvla.pi05_omni_lwc_units_v1 import capture,calibrate,inventory,apply_profile,SOURCE_SHA
from qvla import pi05_omni_lwc_units_v1 as native_adapter

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--capture',type=Path,required=True);p.add_argument('--manifest32',type=Path,required=True);p.add_argument('--cpu-proof',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke-only',action='store_true');p.add_argument('--resume-from',type=Path,required=True);a=p.parse_args();assert not a.smoke_only
assert torch.cuda.is_available();started=time.monotonic()
cpu=json.loads(a.cpu_proof.read_text());assert cpu['gate']=='PASS_PI05_NATIVE_OMNI_LWC_REPLAY_GRADIENT_AND_PACK_CPU' and cpu['source_sha256']==SOURCE_SHA
assert cpu['adapter_sha256']==digest(Path(native_adapter.__file__))
manifest=json.loads(a.manifest32.read_text());assert manifest['gate']=='PASS_SELECTED_32_CALIBRATION_NPZ_AND_ROLE_MANIFESTS' and len(manifest['records'])==32
for row in manifest['records']:assert row['role']=='calibration' and row['resets']==[0] and digest(row['manifest'])==row['sha256']
receipt=json.loads((a.capture/'result.json').read_text());assert receipt['gate']=='PASS_PI05_414_CALIBRATION_REPLAY_STATS' and digest(a.capture/'statistics.pt')==receipt['profile_sha256']
cache=torch.load(a.capture/'statistics.pt',map_location='cpu',weights_only=True);assert cache['backend']==BACKEND_ID
rows=cache['samples'];assert len(rows)==len({r['sha256'] for r in rows})==32
source=a.root/'sources/OmniQuant-archive-v1';identity=checkpoint_identity(a.root/'models/pi05_libero_pytorch_bridge_v2')['sha256']
a.output.mkdir(exist_ok=False,parents=True);(a.output/'units').mkdir()
payload=dict(format='PI05_ORIGINAL_OMNI_LWC_NATIVE_UNITS_V1',backend=BACKEND_ID,model_identity=identity,
             capture_sha256=receipt['profile_sha256'],query_sha256=[r['sha256'] for r in rows],source_sha256=SOURCE_SHA,
             entries=[],full_policy_admitted=False,LET=False,weight_bits=4,activation_bits=16,
             epochs=2 if a.smoke_only else 20,lwc_lr=.01,weight_decay=0,calibration_only=True,formal_episodes=0,
             adaptation='Native attention/MLP unit reconstruction; original LWC, no LET, not complete paper reproduction')
write_json(a.output/'binding.json',{k:v for k,v in payload.items() if k!='entries'})

# Resume only completed immutable independent units, never an unfinished optimizer.
prior=json.loads((a.resume_from/'progress.json').read_text())
assert not (a.resume_from/'result.json').exists()
assert 0 < len(prior['entries']) <=126 and not prior['full_policy_admitted']
for key in payload:
    if key!='entries':assert prior[key]==payload[key], 'Resume recipe/input identity differs: '+key
assert digest(Path(native_adapter.__file__))==cpu['adapter_sha256']
for i,entry in enumerate(prior['entries']):
    assert entry['file']==f'units/{i:03d}.pt'
    src=a.resume_from/entry['file'];assert not src.is_symlink() and digest(src)==entry['sha256']
    artifact=torch.load(src,map_location='cpu',weights_only=True)
    for key,value in dict(epochs=20,learning_rate=.01,weight_decay=0,LET=False,disabled_quant_bitwise=True,gradient_finite_nonzero=True,teacher_unchanged=True,native_unit_reconstruction=True).items():assert artifact[key]==value
    assert artifact['calls']==entry['calls'] and len(artifact['weights'])==entry['targets']
    for key in ['loss_first','loss_last']:assert np.isfinite(artifact[key]) and artifact[key]==entry[key]
    import os
    os.link(src,a.output/entry['file'])  # Immutable packed unit files; no duplicated parameter storage.
    payload['entries'].append(entry)
write_json(a.output/'resume-proof.json',dict(gate='PASS_COMPLETED_INDEPENDENT_UNIT_PREFIX_SHA_AND_RECIPE',parent=str(a.resume_from),progress_sha256=digest(a.resume_from/'progress.json'),units_reused=len(payload['entries']),unfinished_unit_reused=False,parent_unchanged=True))
write_json(a.output/'progress.json',payload)

def action(policy,row):
    assert digest(row['path'])==row['sha256'] and row['role']=='calibration' and row['reset_index']==0
    with np.load(row['path'],allow_pickle=False) as q:
        obs={'observation/image':q['image'],'observation/wrist_image':q['wrist_image'],'observation/state':q['state'],'prompt':str(q['prompt'].item())}
        actual=np.asarray(policy.infer(obs,noise=q['policy_noise'])['actions']).copy();expected=q['raw_policy_chunk'].copy()
    assert actual.shape==(10,7) and np.isfinite(actual).all()
    return actual,expected

def release():
    gc.collect();torch.cuda.empty_cache();assert torch.cuda.memory_allocated()<512*1024**2

def build():
    policy=load_policy(a.root);assert torch.get_float32_matmul_precision()=='high' and torch.backends.cuda.matmul.allow_tf32
    units=inventory(policy._model)
    if a.smoke_only:
        units=[next((name,targets) for name,targets in units if category in name and name.endswith('.'+suffix))
               for category in ['gemma_expert','language_model','vision_tower'] for suffix in ['self_attn','mlp']]
    for index,(name,targets) in enumerate(units):
        if index < len(prior['entries']):
            entry=prior['entries'][index];assert entry['name']==name and entry['targets']==len(targets)
            artifact=torch.load(a.output/entry['file'],map_location='cpu',weights_only=True)
            assert {name+'.'+n for n in artifact['weights']}==set(targets)
            from qvla.pi05_packed_weight_v1 import unpack,weight_sha
            for relative,weight in artifact['weights'].items():
                linear=policy._model.get_submodule(name+'.'+relative)
                value=unpack(weight,torch.device('cpu'),linear.weight.dtype)
                assert value.shape==linear.weight.shape and torch.isfinite(value).all() and weight_sha(value)==weight['weight_sha256']
            del artifact
            continue
        module=policy._model.get_submodule(name);calls,teachers=capture(policy,module,rows)
        artifact=calibrate(module,calls,teachers,source,epochs=payload['epochs'])
        assert {name+'.'+n for n in artifact['weights']}==set(targets)
        file=f'units/{index:03d}.pt';temporary=a.output/(file+'.tmp');torch.save(artifact,temporary);temporary.replace(a.output/file)
        payload['entries'].append(dict(name=name,file=file,sha256=digest(a.output/file),targets=len(targets),calls=len(calls),loss_first=artifact['loss_first'],loss_last=artifact['loss_last']))
        write_json(a.output/'progress.json',payload)
        del calls,teachers,artifact;gc.collect();torch.cuda.empty_cache()
        print('OMNI_LWC_UNIT',index+1,len(units),name,flush=True)
    for row in rows:
        actual,expected=action(policy,row);assert np.array_equal(actual,expected),'Teacher changed during LWC calibration'
    assert all(not p.requires_grad for p in policy._model.parameters())
    del policy


def finish():
    selected=[next(r for r in rows if r['suite']==suite) for suite in sorted({r['suite'] for r in rows})]
    def evaluate():
        policy=load_policy(a.root);assert torch.get_float32_matmul_precision()=='high' and torch.backends.cuda.matmul.allow_tf32
        apply_profile(policy._model,payload,a.output,allow_pending=True)
        values=[];metrics=[]
        for row in selected:
            actual,expected=action(policy,row);repeat,_=action(policy,row);assert np.array_equal(actual,repeat)
            values.append(actual);metrics.append(dict(suite=row['suite'],input_sha256=row['sha256'],rmse_vs_native_bf16=float(np.sqrt(np.mean((actual-expected)**2))),max_abs_vs_native_bf16=float(np.max(np.abs(actual-expected)))))
        assert all(not p.requires_grad for p in policy._model.parameters())
        del policy
        return np.stack(values),metrics
    assert len(payload['entries'])==126 and sum(x['targets'] for x in payload['entries'])==414
    first,metrics=evaluate();release();second,_=evaluate();release();assert np.array_equal(first,second),'Independent full-policy reload actions differ'
    np.savez_compressed(a.output/'calibration-action-checks.npz',quantized=first,reloaded=second)
    payload.update(full_policy_admitted=True,gate='PASS_PI05_OMNI_LWC_NATIVE_414_RELOAD_FINITE')
    torch.save(payload,a.output/'profile.pt')
    return dict(gate=payload['gate'],profile_sha256=digest(a.output/'profile.pt'),quantized_action_differences=metrics,target_units=126,target_projections=414,quantized_repeat_reload_bitwise=True)


torch.cuda.reset_peak_memory_stats()
try:
    build();release()
    result=(dict(gate='PASS_PI05_OMNI_LWC_NATIVE_SIX_UNITS_GPU_SMOKE',full_policy_admitted=False,target_units=6,target_projections=sum(x['targets'] for x in payload['entries']))
            if a.smoke_only else finish())
    result.update(backend=BACKEND_ID,model_identity=identity,teacher_actions_restored_bitwise=True,gradient_finite_nonzero=True,parameters_frozen=True,finite=True,LET=False,
                  source_sha256=SOURCE_SHA,calibration_queries=32,calibration_only=True,formal_episodes=0,epochs=payload['epochs'],unit_reconstruction_not_full_paper=True,
                  wall_seconds_including_load=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),finished_at=now(),not_an_int4_execution_kernel=True)
    write_json(a.output/'result.json',result);print(json.dumps(result))
except BaseException as error:
    write_json(a.output/'failure.json',dict(gate='FAIL_PI05_OMNI_LWC_CALIBRATION',error=repr(error),units_complete=len(payload['entries']),full_policy_admitted=False,formal_episodes=0,wall_seconds=time.monotonic()-started))
    raise
