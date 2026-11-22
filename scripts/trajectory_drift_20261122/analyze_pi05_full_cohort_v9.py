"""Full five-initial same-backend cohort audit; skip actual live stages before raw reads."""
from pathlib import Path
import argparse
import importlib
import json
import os
import sys
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--control',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--code',type=Path,required=True)
p.add_argument('--conditions',nargs='+',default=['bf16','gptq_w4a16','awq_w4a16','omniquant_w4a16','smoothquant_w4a8','duquant_w4a8','duquant_w4a4'])
a=p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(a.code))
from qvla.infra.common import digest
from qvla.trajectory_diagnostics_v1 import rotation_difference
backend='openpi-pytorch-original-independent-h10-v2'
conditions=a.conditions
assert conditions[0]=='bf16' and len(set(conditions))==len(conditions)
assert 'smoothquant_w4a4' not in conditions, 'Cancelled configuration must remain separate'
resets=[5,15,25,35,45]
suites=['libero_spatial','libero_object','libero_goal','libero_10']
active_stages=[]
for receipt in (a.root/'control').glob('*/stages/*/launch-receipt.json'):
    item=json.loads(receipt.read_text());stage=receipt.parent.resolve()
    proc=Path('/proc',str(item['supervisor_pid']),'cmdline')
    if proc.exists() and str(stage).encode() in proc.read_bytes():
        active_stages.append(stage)
# Ignore historical RUNNING labels: only actual owned supervisors define live stages.
a.output.mkdir(exist_ok=False,parents=True)
records={};private=[];poses={};queries={};examples=[]
paths=[*(a.root/'rounds').glob('*/case-runs/*/result.json'),*(a.root/'control').glob('*/stages/*/case-runs/*/result.json')]
for file in paths:
    if any(file.is_relative_to(stage) for stage in active_stages):continue
    result=json.loads(file.read_text())
    if (result.get('backend')!=backend or result.get('role')!='development' or
            result.get('reset_index') not in resets or result.get('condition') not in conditions or
            not result.get('gate','').startswith('PASS_PI05_TORCH_')):
        continue
    c=result['condition'];key=(c,result['suite'],result['task_id'],result['reset_index'])
    assert key not in records,'Repeated same-backend scientific case'
    folder=file.parent;stage=folder.parent.parent
    plan=stage/'case-plans'/(result['round_id']+'-runtime.json')
    module={'bf16':'qvla.audit_pi05_torch_trace_v2',
            'duquant_w4a8':'qvla.audit_pi05_torch_duquant_trace_v1','duquant_w4a4':'qvla.audit_pi05_torch_duquant_trace_v1',
            'omniquant_w4a16':'qvla.audit_pi05_torch_omni_trace_v1','awq_w4a16':'qvla.audit_pi05_torch_awq_trace_v1','gptq_w4a16':'qvla.audit_pi05_torch_gptq_trace_v1'}.get(c,'qvla.audit_pi05_torch_sq_trace_v2')
    proof=importlib.import_module(module).audit(folder,plan)
    private.append(dict(result=str(file),result_sha256=digest(file),audit=proof,current_round=folder.is_relative_to(a.control)))
    positions=[];quaternions=[];initial_position=None
    for step in sorted(folder.glob('step-*.npz')):
        with np.load(step,allow_pickle=False) as raw:
            if initial_position is None:initial_position=raw['robot0_eef_pos'].copy()
            positions.append(raw['post_robot0_eef_pos'].copy())
            quaternions.append(raw['post_robot0_eef_quat'].copy())
    poses[key]=(np.array(positions),np.array(quaternions))
    if result['task_id']==0 and result['reset_index']==5:
        examples.append(dict(condition=c,suite=result['suite'],task_id=0,reset_index=5,success=bool(result['success']),position_m=np.vstack([initial_position,positions]).tolist()))
    with np.load(folder/'query-00000.npz',allow_pickle=False) as raw:
        queries[key]={k:raw[k].copy() for k in ['image','wrist_image','state','prompt','policy_noise','raw_policy_chunk']}
    records[key]=dict(condition=c,suite=result['suite'],task_id=result['task_id'],reset_index=result['reset_index'],
                      success=bool(result['success']),control_steps=result['control_steps'],
                      policy_queries=result['policy_queries'],
                      wall_seconds_including_verification_load=result.get('wall_s_including_verification_and_load'),
                      wall_seconds_environment_and_queries=result.get('wall_s_env_and_queries'))
assert all(sum(k[0]==c for k in records)==200 for c in conditions), 'Selected conditions must each contain all 200 unique cases'
pairs=[];traces={};vectors={}
for (c,s,t,r),row in records.items():
    if c=='bf16':continue
    refkey=('bf16',s,t,r);assert refkey in records
    for name in ['image','wrist_image','state','prompt','policy_noise']:
        assert np.array_equal(queries[(c,s,t,r)][name],queries[refkey][name])
    pos,quat=poses[(c,s,t,r)];rp,rq=poses[refkey];n=min(len(pos),len(rp))
    trans_vector=(pos[:n]-rp[:n])*1000
    rotation_vector=np.array([rotation_difference(x,y)*180/np.pi for x,y in zip(quat[:n],rq[:n])])
    vectors[c,s,t,r]=(trans_vector,rotation_vector)
    trans=np.linalg.norm(trans_vector,axis=1)
    rot=np.linalg.norm(rotation_vector,axis=1)
    assert np.isfinite(trans).all() and np.isfinite(rot).all()
    delta=queries[(c,s,t,r)]['raw_policy_chunk'][:5]-queries[refkey]['raw_policy_chunk'][:5]
    pairs.append(dict(row,reference_success=records[refkey]['success'],common_steps=n,
                      position_p95_mm=float(np.percentile(trans,95)),rotation_p95_deg=float(np.percentile(rot,95)),
                      first_executed5_translation_rmse_native_units=float(np.sqrt(np.mean(delta[:,:3]**2))),
                      first_executed5_rotation_rmse_native_units=float(np.sqrt(np.mean(delta[:,3:6]**2)))))
    traces[c,s,t,r]=(trans,rot)
coverage={};missing=[]
for c in conditions:
    rows=[x for (condition,_,_,_),x in records.items() if condition==c]
    coverage[c]=dict(complete=len(rows),planned_full_cohort=200,success=sum(x['success'] for x in rows),
                     suites={s:dict(complete=sum(x['suite']==s for x in rows),success=sum(x['suite']==s and x['success'] for x in rows)) for s in suites})
    missing += [dict(condition=c,suite=s,task_id=t,reset_index=r) for s in suites for t in range(10) for r in resets if (c,s,t,r) not in records]
summary=dict(gate='PASS_COMPLETE_200_PI05_SAME_BACKEND_RAW_GEOMETRY_AUDIT',backend=backend,
             native_jax_combined=False,active_other_initial_stages_skipped=len(active_stages),user_cancelled_conditions=['smoothquant_w4a4'],reset_indices=resets,suites=suites,development_tasks=40,
             complete_unique=len(records),paired_quantized=len(pairs),coverage=coverage,missing=missing,
             records=list(records.values()),pairs=pairs,
             qualification='Same backend and first input/noise. Closed-loop separation includes feedback. Unequal coverage cannot be ranked as complete methods. No shared changepoint or OPD efficacy established.')
(a.output/'summary-public.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
curves=[]
for c in conditions[1:]:
    for s in suites:
        available=[v for (cc,ss,_,_),v in traces.items() if cc==c and ss==s]
        if not available:continue
        length=max(len(x[0]) for x in available)
        row=dict(condition=c,suite=s,step=list(range(1,length+1)))
        for ai,name in enumerate(['position_mm','rotation_deg']):
            grid=np.full((len(available),length),np.nan)
            for j,x in enumerate(available):grid[j,:len(x[ai])]=x[ai]
            count=np.isfinite(grid).sum(0);med=np.nanmedian(grid,axis=0);lo,hi=np.nanpercentile(grid,[25,75],axis=0)
            clean=lambda v:[float(x) if count[i]>=5 else None for i,x in enumerate(v)]
            row[name]=dict(n=count.tolist(),median=clean(med),p25=clean(lo),p75=clean(hi))
        curves.append(row)
(a.output/'curves-public.json').write_text(json.dumps(curves,allow_nan=False))
(a.output/'examples-private.json').write_text(json.dumps(examples,allow_nan=False))
(a.output/'audit-private.json').write_text(json.dumps(dict(cases=private),indent=2))


# Decompose cross-case error vectors, not same-state model noise or causality.
decomposition=[]
for c in conditions[1:]:
    for suite in suites:
        selected=[v for (cc,ss,_,_),v in vectors.items() if cc==c and ss==suite]
        if not selected:continue
        length=max(len(v[0]) for v in selected)
        out=dict(condition=c,suite=suite,step=list(range(1,length+1)))
        for axis,field in enumerate(['position_world_mm','rotation_world_log_deg']):
            grid=np.full((len(selected),length,3),np.nan)
            for j,v in enumerate(selected):grid[j,:len(v[axis])]=v[axis]
            count=np.isfinite(grid[:,:,0]).sum(axis=0)
            mean=np.nanmean(grid,axis=0)
            bias=np.linalg.norm(mean,axis=1)
            dispersion=np.sqrt(np.nanmean(np.sum((grid-mean[None,:,:])**2,axis=2),axis=0))
            total=np.sqrt(np.nanmean(np.sum(grid**2,axis=2),axis=0))
            assert np.allclose(total**2,bias**2+dispersion**2,rtol=1e-12,atol=1e-8)
            clean=lambda values:[float(v) if count[i]>=5 else None for i,v in enumerate(values)]
            out[field]=dict(n=count.tolist(),mean_vector=[clean(mean[:,k]) for k in range(3)],center_magnitude=clean(bias),rms_dispersion=clean(dispersion),total_rms=clean(total))
        decomposition.append(out)
(a.output/'error-decomposition-public.json').write_text(json.dumps(dict(gate='PASS_PAIRED_VECTOR_CENTER_SPREAD_RMS_IDENTITY',curves=decomposition,qualification='Cross-case closed-loop displacement vectors, same control step/world frame; unequal trajectory endings change composition. Rotation uses world-frame principal SO(3) log, not Euler angles or a global Frechet mean. No same-state noise mechanism, common changepoint, or OPD efficacy proved.'),allow_nan=False))

print(json.dumps(dict(gate=summary['gate'],complete_unique=len(records),coverage=coverage,decomposition_written=True)),flush=True)
