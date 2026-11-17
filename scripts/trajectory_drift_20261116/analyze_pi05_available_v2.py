"""Terminal-only multi-round, same-PyTorch-backend raw audits and descriptive pose separation."""
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
a=p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(a.root/'code-084-cpu-prepared-20261112-v1'))
from qvla.infra.common import digest
from qvla.trajectory_diagnostics_v1 import rotation_difference
backend='openpi-pytorch-original-independent-h10-v2'
conditions=['bf16','smoothquant_w4a8','smoothquant_w4a4','gptq_w4a16']
suites=['libero_spatial','libero_object','libero_goal','libero_10']
for stage in (a.control/'stages').iterdir():
    if (stage/'status.json').exists():
        assert json.loads((stage/'status.json').read_text())['state'] not in ['RUNNING','PREFLIGHT']
a.output.mkdir(exist_ok=False,parents=True)
records={};private=[];poses={};queries={}
paths=[*(a.root/'rounds').glob('*/case-runs/*/result.json'),*(a.root/'control').glob('*/stages/*/case-runs/*/result.json')]
for file in paths:
    result=json.loads(file.read_text())
    if (result.get('backend')!=backend or result.get('role')!='development' or
            result.get('reset_index')!=5 or result.get('condition') not in conditions or
            not result.get('gate','').startswith('PASS_PI05_TORCH_')):
        continue
    c=result['condition'];key=(c,result['suite'],result['task_id'])
    assert key not in records,'Repeated same-backend scientific case'
    folder=file.parent;stage=folder.parent.parent
    plan=stage/'case-plans'/(result['round_id']+'-runtime.json')
    module={'bf16':'qvla.audit_pi05_torch_trace_v2',
            'gptq_w4a16':'qvla.audit_pi05_torch_gptq_trace_v1'}.get(c,'qvla.audit_pi05_torch_sq_trace_v2')
    proof=importlib.import_module(module).audit(folder,plan)
    private.append(dict(result=str(file),result_sha256=digest(file),audit=proof,current_round=folder.is_relative_to(a.control)))
    positions=[];quaternions=[]
    for step in sorted(folder.glob('step-*.npz')):
        with np.load(step,allow_pickle=False) as raw:
            positions.append(raw['post_robot0_eef_pos'].copy())
            quaternions.append(raw['post_robot0_eef_quat'].copy())
    poses[key]=(np.array(positions),np.array(quaternions))
    with np.load(folder/'query-00000.npz',allow_pickle=False) as raw:
        queries[key]={k:raw[k].copy() for k in ['image','wrist_image','state','prompt','policy_noise','raw_policy_chunk']}
    records[key]=dict(condition=c,suite=result['suite'],task_id=result['task_id'],reset_index=5,
                      success=bool(result['success']),control_steps=result['control_steps'],
                      policy_queries=result['policy_queries'],
                      wall_seconds_including_verification_load=result.get('wall_s_including_verification_and_load'),
                      wall_seconds_environment_and_queries=result.get('wall_s_env_and_queries'))
assert sum(k[0]=='bf16' for k in records)==40
pairs=[];traces={}
for (c,s,t),row in records.items():
    if c=='bf16':continue
    refkey=('bf16',s,t);assert refkey in records
    for name in ['image','wrist_image','state','prompt','policy_noise']:
        assert np.array_equal(queries[(c,s,t)][name],queries[refkey][name])
    pos,quat=poses[(c,s,t)];rp,rq=poses[refkey];n=min(len(pos),len(rp))
    trans=np.linalg.norm(pos[:n]-rp[:n],axis=1)*1000
    rot=np.array([np.linalg.norm(rotation_difference(x,y))*180/np.pi for x,y in zip(quat[:n],rq[:n])])
    assert np.isfinite(trans).all() and np.isfinite(rot).all()
    delta=queries[(c,s,t)]['raw_policy_chunk'][:5]-queries[refkey]['raw_policy_chunk'][:5]
    pairs.append(dict(row,reference_success=records[refkey]['success'],common_steps=n,
                      position_p95_mm=float(np.percentile(trans,95)),rotation_p95_deg=float(np.percentile(rot,95)),
                      first_executed5_translation_rmse_native_units=float(np.sqrt(np.mean(delta[:,:3]**2))),
                      first_executed5_rotation_rmse_native_units=float(np.sqrt(np.mean(delta[:,3:6]**2)))))
    traces[c,s,t]=(trans,rot)
coverage={};missing=[]
for c in conditions:
    rows=[x for (condition,_,_),x in records.items() if condition==c]
    coverage[c]=dict(complete=len(rows),planned_first_initial=40,success=sum(x['success'] for x in rows),
                     suites={s:dict(complete=sum(x['suite']==s for x in rows),success=sum(x['suite']==s and x['success'] for x in rows)) for s in suites})
    missing += [dict(condition=c,suite=s,task_id=t,reset_index=5) for s in suites for t in range(10) if (c,s,t) not in records]
summary=dict(gate='PASS_AVAILABLE_PI05_SAME_BACKEND_RAW_GEOMETRY_AUDIT',backend=backend,
             native_jax_combined=False,reset_indices=[5],suites=suites,development_tasks=40,
             complete_unique=len(records),paired_quantized=len(pairs),coverage=coverage,missing=missing,
             records=list(records.values()),pairs=pairs,
             qualification='Same backend and first input/noise. Closed-loop separation includes feedback. Unequal coverage cannot be ranked as complete methods. No shared changepoint or OPD efficacy established.')
(a.output/'summary-public.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
curves=[]
for c in conditions[1:]:
    for s in suites:
        available=[v for (cc,ss,_),v in traces.items() if cc==c and ss==s]
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
(a.output/'audit-private.json').write_text(json.dumps(dict(cases=private),indent=2))
print(json.dumps(dict(gate=summary['gate'],complete_unique=len(records),coverage=coverage)))
