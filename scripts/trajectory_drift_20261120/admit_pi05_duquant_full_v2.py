"""Original DuQuant full pi05 binding: fixed calibration32, finite/reload gate.

No development episodes. The original online rotation initializes on the first
actual call per projection and then freezes, as in upstream QuantLinear.
"""
import argparse,gc,json,os,random,shutil,time
from pathlib import Path
import numpy as np
import torch
from qvla.infra.common import digest,write_json,now
from qvla.pi05_torch_backend_v2 import load_policy,BACKEND_ID
from qvla.pi05_duquant_native_v2 import original_quantizer,attach,set_mode,attach_frozen_rotations_for_binding_check


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--activation-bits',type=int,choices=[4,8],required=True)
    p.add_argument('--rotation-parent',type=Path,required=True)
    p.add_argument('--audit-suite');a=p.parse_args();out=a.stage/'profile'
    if a.audit_suite:
        proof=json.loads((out/'result.json').read_text())
        assert proof['gate']=='PASS_PI05_ORIGINAL_DUQUANT_414_FP32_BINDING_RELOAD_FINITE_TRANSFORM'
        assert proof['activation_bits']==a.activation_bits and a.audit_suite in [x['suite'] for x in proof['unquantized_transform_checks']]
        assert digest(out/'profile.pt')==proof['profile_sha256']
        print(json.dumps(dict(gate='PASS_DUQUANT_CALIBRATION_SUITE_EVIDENCE',suite=a.audit_suite,formal_episodes=0)))
        return
    started=time.monotonic();out.mkdir(exist_ok=False);assert torch.cuda.is_available()
    torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
    src=a.root/'sources/DuQuant-archive-v1'
    ready=json.loads((a.root/'preparations/admission-first-20261119-v1/readiness/readiness.json').read_text())
    index={r['path']:r['sha256'] for source in ready['source_directories'] for r in source.get('files',[])}
    source_sha={name:index[str((src/name).relative_to(a.root))] for name in ['quantize/quantizer.py','quantize/int_linear.py','quantize/const.py','utils.py']}
    assets=a.root/'preparations/pi05-duquant-20261112-v3'
    assert digest(assets/'Rot.pkl')==json.loads((assets/'result.json').read_text())['rotation_asset_sha256']
    shutil.copyfile(assets/'Rot.pkl',out/'Rot.pkl');os.chdir(out)
    quantizer=original_quantizer(src,source_sha)
    capture=a.root/'rounds/20261111-084-pi05-smoothquant-a8-v2/capture'
    assert digest(capture/'statistics.pt')==json.loads((capture/'result.json').read_text())['profile_sha256']
    cache=torch.load(capture/'statistics.pt',map_location='cpu',weights_only=True);rows=cache['samples']
    assert len(rows)==32 and len({x['sha256'] for x in rows})==32
    selected=[next(x for x in rows if x['suite']==s) for s in ['libero_spatial','libero_object','libero_goal','libero_10']]
    frozen=torch.load(a.rotation_parent,map_location='cpu',weights_only=True)
    assert frozen['format']=='PI05_ORIGINAL_DUQUANT_NATIVE_V1' and len(frozen['projections'])==414 and frozen['finite_calibration_passed']
    assert frozen['activation_bits']==a.activation_bits and frozen['calibration_query_sha256']==[r['sha256'] for r in rows]
    assert frozen['source_sha256']==source_sha
    calls=0
    def action(policy,row):
        nonlocal calls
        assert digest(row['path'])==row['sha256']
        with np.load(row['path'],allow_pickle=False) as sample:
            obs={'observation/image':sample['image'],'observation/wrist_image':sample['wrist_image'],
                 'observation/state':sample['state'],'prompt':str(sample['prompt'].item())}
            noise=sample['policy_noise'].copy();expected=sample['raw_policy_chunk'].copy()
        result=np.asarray(policy.infer(obs,noise=noise)['actions']).copy();calls+=1
        assert result.shape==(10,7) and np.isfinite(result).all()
        return result,expected
    def release():
        gc.collect();torch.cuda.empty_cache()
        assert torch.cuda.memory_allocated()<512*1024**2,'Previous full policy still allocated'
    def build(payload=None):
        policy=load_policy(a.root)
        torch.manual_seed(7);random.seed(7);np.random.seed(7)
        modules=attach_frozen_rotations_for_binding_check(policy._model,quantizer,a.activation_bits,frozen['projections']) if payload is None else attach(policy._model,quantizer,a.activation_bits,payload)
        assert len(modules)==414 and all(not x.requires_grad for x in policy._model.parameters())
        return policy,modules
    # First teacher query proves unchanged inherited baseline before rotation.
    native=load_policy(a.root);value,expected=action(native,rows[0]);assert np.array_equal(value,expected)
    del native;release()
    policy,modules=build();outputs=[]
    for i,row in enumerate(rows):
        value,expected=action(policy,row);outputs.append(value)
        write_json(out/'progress.json',dict(phase='fixed_rotation_validation',queries_complete=i+1,calibration_queries=32,
            initialized_projections=sum(bool(x.act.init_duquant_params) for x in modules.values()),formal_episodes=0))
    projections={name:layer.record() for name,layer in modules.items()}
    for name,row in projections.items():
        before=frozen['projections'][name]
        assert torch.equal(row['R'],before['R']) and ((row['permutation_list'] is None and before['permutation_list'] is None) or torch.equal(row['permutation_list'],before['permutation_list'])),'Frozen rotations changed'
    payload=dict(format='PI05_ORIGINAL_DUQUANT_NATIVE_FP32_BINDING_V2',backend=BACKEND_ID,activation_bits=a.activation_bits,
        weight_bits=4,projections=projections,source_sha256=source_sha,rotation_steps=1024,permutation_times=1,
        rotation_asset_sha256=digest(out/'Rot.pkl'),calibration_query_sha256=[r['sha256'] for r in rows],
        base_weights_in_parent_checkpoint=True,finite_calibration_passed=True,rotation_parent_sha256=digest(a.rotation_parent),rotations_reused_bitwise=True)
    torch.save(payload,out/'profile.pt')
    quantized=[]
    for row in selected:quantized.append(action(policy,row)[0])
    # Disable quantization using the SAME learned transforms, keeping native
    # execution dtype. Do not relax this preregistered action-level gate.
    set_mode(modules,False);gc.collect();torch.cuda.empty_cache()
    metrics=[];unquantized=[]
    for row in selected:
        value,expected=action(policy,row);error=value-expected
        rmse=float(np.sqrt(np.mean(error**2)));maximum=float(np.max(np.abs(error)))
        metrics.append(dict(suite=row['suite'],rmse=rmse,max_abs=maximum))
        unquantized.append(value)
        write_json(out/'unquantized-transform-checks.json',dict(checks=metrics,rmse_limit=.002,max_abs_limit=.02))
        assert rmse<=.002 and maximum<=.02,'Unquantized full-policy paired rotation gate rejected'
    del modules,policy;release()
    payload=torch.load(out/'profile.pt',map_location='cpu',weights_only=True)
    reloaded,modules=build(payload);restored=[]
    for i,row in enumerate(selected):
        value,_=action(reloaded,row);assert np.array_equal(value,quantized[i]),'Full DuQuant action reload mismatch'
        restored.append(value)
    assert all(not p.requires_grad for p in reloaded._model.parameters())
    for layer in modules.values():layer.record()
    del layer,modules,reloaded;release()
    native=load_policy(a.root)
    for row in selected:
        value,expected=action(native,row);assert np.array_equal(value,expected),'Teacher did not restore'
    del native;release()
    np.savez_compressed(out/'calibration-actions.npz',quantized=np.stack(quantized),
        unquantized_transformed=np.stack(unquantized),reloaded=np.stack(restored))
    result=dict(gate='PASS_PI05_ORIGINAL_DUQUANT_414_FP32_BINDING_RELOAD_FINITE_TRANSFORM',backend=BACKEND_ID,
        activation_bits=a.activation_bits,weight_bits=4,profile_sha256=digest(out/'profile.pt'),
        projections=414,calibration_queries=32,full_policy_admitted=True,formal_episodes=0,
        source_sha256=source_sha,unquantized_transform_checks=metrics,
        independent_reload_actions_bitwise=True,base_frozen=True,native_teacher_restored_bitwise=True,
        policy_queries_including_validation=calls,wall_seconds=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),finished_at=now(),
        rotation_parent_sha256=digest(a.rotation_parent),rotations_reused_bitwise=True,
        implementation='Original DuQuant operations; FP32 internal paired products and native output dtype; runtime representation fix, not changed rotations/quantizer/data; no INT4 kernel claim')
    write_json(out/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))


if __name__=='__main__':main()
