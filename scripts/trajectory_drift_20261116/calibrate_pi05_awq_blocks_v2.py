"""Calibration-only original AWQ port, one block at a time, 32 fixed queries.

No benchmark labels, checkpoint selection, training or development episodes.
Partial block artifacts are never an admitted full-policy profile.
"""
import argparse,gc,json,time,os
from pathlib import Path
import numpy as np
import torch
from qvla.infra.common import digest,canonical,write_json,now
from qvla.baseline_contract import checkpoint_identity
from qvla.pi05_torch_backend_v2 import load_policy,target_names,BACKEND_ID
from qvla.pi05_awq_capture_block_v2 import collect
from qvla.pi05_awq_block_calibration_v1 import calibrate_block
from qvla.pi05_awq_profile_v1 import apply_profile

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--capture',type=Path,required=True);p.add_argument('--manifest32',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--reuse-completed',type=Path);p.add_argument('--layout-proof',type=Path);a=p.parse_args()
started=time.monotonic();assert torch.cuda.is_available()
source=a.root/'sources/quant-source-sparse-v2/llm-awq/awq/quantize'
receipt=json.loads((a.capture/'result.json').read_text());assert receipt['gate']=='PASS_PI05_414_CALIBRATION_REPLAY_STATS'
assert digest(a.capture/'statistics.pt')==receipt['profile_sha256']
index=json.loads(a.manifest32.read_text());assert index['gate']=='PASS_SELECTED_32_CALIBRATION_NPZ_AND_ROLE_MANIFESTS'
assert len(index['records'])==32
for r in index['records']:assert r['role']=='calibration' and r['resets']==[0] and digest(r['manifest'])==r['sha256']
cache=torch.load(a.capture/'statistics.pt',map_location='cpu',weights_only=True);assert cache['backend']==BACKEND_ID
rows=cache['samples'];assert len(rows)==32 and len({r['sha256'] for r in rows})==32
config=json.loads((a.root/'rounds/20261111-084-pi05-smoothquant-a8-v2/config.json').read_text())
identity=checkpoint_identity(a.root/'models/pi05_libero_pytorch_bridge_v2')['sha256'];assert identity==config['model']['identity_sha256']
binding=dict(backend=BACKEND_ID,method='awq_w4a16_native_blocks_v1',model_identity=identity,capture_sha256=receipt['profile_sha256'],manifest32_sha256=digest(a.manifest32),query_sha256=[r['sha256'] for r in rows],code_sha256={str(x.relative_to(Path(__file__).resolve().parents[2])):digest(x) for x in [Path(__file__).resolve(),*[Path(__file__).resolve().parents[2]/'qvla'/n for n in ['pi05_awq_profile_v1.py','pi05_awq_capture_block_v2.py','pi05_awq_block_calibration_v1.py','pi05_awq_replay_batch_v1.py','pi05_awq_native_components_v2.py','pi05_block_replay_v2.py','pi05_packed_weight_v1.py']] ]})
if a.resume:
    prior=json.loads((a.output/'progress.json').read_text());assert prior['binding_sha256']==canonical(binding)
else:
    a.output.mkdir(exist_ok=False,parents=True);(a.output/'blocks').mkdir();prior=dict(binding,binding_sha256=canonical(binding),entries=[])
if a.reuse_completed:
    assert not a.resume and a.layout_proof is not None
    proof=json.loads(a.layout_proof.read_text())
    assert proof['gate']=='PASS_AWQ_CACHE_LAYOUT_VALUES_AND_QUANT_ARTIFACTS_BITWISE_CPU'
    assert proof['production_cache_budget_bytes']==2*1024**3 and proof['calibration_queries_unchanged']==32 and proof['original_algorithm_unchanged']
    legacy=json.loads((a.reuse_completed/'progress.json').read_text())
    old_binding={k:v for k,v in legacy.items() if k not in ['entries','binding_sha256']}
    assert canonical(old_binding)==legacy['binding_sha256']
    for key in ['backend','method','model_identity','capture_sha256','manifest32_sha256','query_sha256']:
        assert legacy[key]==binding[key], 'Changed calibration identity; no artifact reuse'
    for path,sha in legacy['code_sha256'].items():
        if path.endswith('pi05_awq_capture_block_v1.py') or path.startswith('scripts/'):continue
        assert binding['code_sha256'][path]==sha, 'Quantization arithmetic changed; no artifact reuse'
    for i,row in enumerate(legacy['entries']):
        assert row['file']==f'blocks/{i:03d}.pt'
        old=a.reuse_completed/row['file'];new=a.output/row['file']
        assert not old.is_symlink() and digest(old)==row['sha256'] and not new.exists()
        os.link(old,new);assert digest(new)==row['sha256']
    prior['entries']=legacy['entries']
    write_json(a.output/'completed-block-reuse.json',dict(gate='PASS_SAME_RECIPE_COMPLETED_BLOCK_ARTIFACT_SHA_REUSE',source_progress_sha256=digest(a.reuse_completed/'progress.json'),layout_proof_sha256=digest(a.layout_proof),blocks=len(prior['entries']),quantization_arithmetic_unchanged=True,calibration_inputs_unchanged=True))
    write_json(a.output/'progress.json',prior)
write_json(a.output/'binding.json',binding)

def action(policy,row):
    assert digest(row['path'])==row['sha256']
    with np.load(row['path'],allow_pickle=False) as q:
        obs={'observation/image':q['image'],'observation/wrist_image':q['wrist_image'],'observation/state':q['state'],'prompt':str(q['prompt'].item())};noise=q['policy_noise'].copy();expected=q['raw_policy_chunk'].copy()
    value=np.asarray(policy.infer(obs,noise=noise)['actions']).copy()
    assert value.shape==(10,7) and np.isfinite(value).all()
    return value,expected

def build():
    policy=load_policy(a.root);assert torch.get_float32_matmul_precision()=='high' and torch.backends.cuda.matmul.allow_tf32
    names=sorted({n.rsplit('.',2)[0] for n in target_names(policy._model)});assert len(names)==63
    assert [r['name'] for r in prior['entries']]==names[:len(prior['entries'])]
    for row in prior['entries']:assert digest(a.output/row['file'])==row['sha256']
    for i,name in enumerate(names):
        if i<len(prior['entries']):continue
        block=policy._model.get_submodule(name)
        calls,features=collect(policy,block,rows)
        artifact=calibrate_block(block,calls,features,source)
        filename=f'blocks/{i:03d}.pt';path=a.output/filename;assert not path.exists()
        torch.save(artifact,path)
        prior['entries'].append(dict(name=name,file=filename,sha256=digest(path),targets=len(artifact['weights'])))
        write_json(a.output/'progress.json',prior)
        del artifact,calls,features;gc.collect();torch.cuda.empty_cache()
    assert sum(r['targets'] for r in prior['entries'])==414
    for row in rows:
        value,expected=action(policy,row);assert np.array_equal(value,expected),'Teacher did not restore after streamed calibration'
    del policy

selected=[]
for suite in sorted({r['suite'] for r in rows}):selected.append(next(r for r in rows if r['suite']==suite))
assert len(selected)==4
def evaluate(mode):
    policy=load_policy(a.root)
    assert torch.get_float32_matmul_precision()=='high' and torch.backends.cuda.matmul.allow_tf32
    apply_profile(policy._model,prior,a.output,source,scales_only=mode=='scales_only',allow_pending=True)
    outputs=[];metrics=[]
    for row in selected:
        value,expected=action(policy,row);repeat,_=action(policy,row)
        assert np.array_equal(value,repeat),'Quantized/scaled inference is not repeatable'
        outputs.append(value)
        metrics.append(dict(suite=row['suite'],input_sha256=row['sha256'],rmse_vs_native_bf16=float(np.sqrt(np.mean((value-expected)**2))),max_abs_vs_native_bf16=float(np.max(np.abs(value-expected)))))
    del policy
    return np.stack(outputs),metrics

def released():
    gc.collect();torch.cuda.empty_cache()
    assert torch.cuda.memory_allocated()<512*1024**2,'A prior full policy remains allocated'

torch.cuda.reset_peak_memory_stats();build();released()
scaled,scaled_metrics=evaluate('scales_only');released()
first,quant_metrics=evaluate('quantized');released()
second,_=evaluate('quantized');released()
assert np.array_equal(first,second),'Full AWQ serialization/reload action gate failed'
np.savez_compressed(a.output/'calibration-action-checks.npz',scales_only=scaled,quantized=first,reloaded=second)
profile=dict(prior,full_policy_admitted=True,gate='PASS_PI05_AWQ_NATIVE_414_RELOAD_FINITE')
torch.save(profile,a.output/'profile.pt')
result=dict(gate=profile['gate'],backend=BACKEND_ID,profile_sha256=digest(a.output/'profile.pt'),model_identity=identity,calibration_queries=32,calibration_only=True,formal_episodes=0,target_blocks=63,target_projections=414,original_source_algorithms=True,teacher_actions_restored_bitwise=True,quantized_repeat_reload_bitwise=True,parameters_frozen=True,finite=True,scales_only_action_differences=scaled_metrics,quantized_action_differences=quant_metrics,scales_only_not_assumed_bitwise_equivalent=True,not_an_int4_execution_kernel=True,wall_seconds_including_load=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),finished_at=now())
write_json(a.output/'result.json',result);print(json.dumps(result))
