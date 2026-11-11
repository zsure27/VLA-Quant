"""Descriptive paired physical-step drift; missing/unfinished trajectories stay missing."""
import argparse
import json
from pathlib import Path

import numpy as np


def read_trace(directory):
    report=json.loads((directory/'result.json').read_text())
    if report.get('gate')!='PASS_GROOT_PAIRED_TRACE':raise ValueError('Incomplete trajectory '+str(directory))
    positions=[];rotations=[];actions=[]
    for path in sorted(directory.glob('query-*.npz')):
        with np.load(path,allow_pickle=False) as row:
            positions.append(row['post_robot0_eef_pos'])
            rotations.append(row['post_robot0_eef_quat'])
            actions.append(row['action'])
    if len(positions)!=report['queries']:raise ValueError('Trace/query count differs')
    return report,np.asarray(positions),np.asarray(rotations),np.asarray(actions)


def drift(left,right):
    n=min(len(left[1]),len(right[1]))
    delta=(right[1][:n]-left[1][:n])*1000
    lq=left[2][:n]/np.linalg.norm(left[2][:n],axis=1,keepdims=True)
    rq=right[2][:n]/np.linalg.norm(right[2][:n],axis=1,keepdims=True)
    angle=np.degrees(2*np.arccos(np.clip(np.abs(np.sum(lq*rq,axis=1)),0,1)))
    return dict(physical_control_step=list(range(1,n+1)),translation_error_mm=np.linalg.norm(delta,axis=1).tolist(),
                signed_translation_delta_mm=delta.tolist(),rotation_geodesic_deg=angle.tolist(),
                observed_common_prefix_steps=n,reference_steps=len(left[1]),candidate_steps=len(right[1]),
                no_padding=True,no_time_warping=True)


def first_window(mask,persistence=5):
    for i in range(len(mask)-persistence+1):
        if all(mask[i:i+persistence]):return i+1
    return None


def summarize(root):
    suites=['libero_spatial','libero_object','libero_goal','libero_10'];controls={};pairs=[];coverage=[]
    for suite in suites:
        directory=root/suite
        try:
            baseline=read_trace(directory/'bf16')
            repeat=read_trace(directory/'bf16-repeat')
        except (FileNotFoundError,ValueError) as error:
            coverage.append(dict(suite=suite,state='reference_or_repeat_incomplete',error=str(error)));continue
        controls[suite]=drift(baseline,repeat)
        for method in ['smoothquant-w4a8','smoothquant-w4a4']:
            try:candidate=read_trace(directory/method)
            except (FileNotFoundError,ValueError):
                coverage.append(dict(suite=suite,condition=method,state='not_complete'));continue
            item=drift(baseline,candidate)
            item.update(suite=suite,condition=method,task_id=0,reset_index=5,
                        baseline_success=baseline[0]['success'],candidate_success=candidate[0]['success'],
                        reference_wall_seconds=baseline[0]['wall_seconds_including_admission'],
                        candidate_wall_seconds=candidate[0]['wall_seconds_including_admission'])
            pairs.append(item)
    # The floor is determined only from BF16 repeat controls, never candidate errors.
    all_t=[v for row in controls.values() for v in row['translation_error_mm']]
    all_r=[v for row in controls.values() for v in row['rotation_geodesic_deg']]
    if all_t and all_r:
        thresholds=dict(translation_mm=max(5.,5*float(np.percentile(all_t,99))),
                        rotation_deg=max(2.,5*float(np.percentile(all_r,99))),persistence_control_steps=5,
                        source='BF16 repeat matched physical steps only; fixed rule before candidate comparison')
        for item in pairs:
            t=np.asarray(item['translation_error_mm']);r=np.asarray(item['rotation_geodesic_deg'])
            item['first_persistent_translation_window_step']=first_window(t>thresholds['translation_mm'])
            item['first_persistent_rotation_window_step']=first_window(r>thresholds['rotation_deg'])
            item['translation_bias_mm']=np.mean(item['signed_translation_delta_mm'],axis=0).tolist()
            item['translation_error_p50_p90_mm']=np.percentile(t,[50,90]).tolist()
            item['rotation_error_p50_p90_deg']=np.percentile(r,[50,90]).tolist()
    else:thresholds=None
    return dict(schema='GROOT_DESCRIPTIVE_PAIRED_DRIFT_V1',planned_suite_count=4,task_ids=[0],reset_indices=[5],
                full_40_task_5_initial_study=False,paired_trajectories=len(pairs),
                thresholds=thresholds,repeat_controls=controls,pairs=pairs,missing_coverage=coverage,
                interpretation='Pilot descriptive windows; no compensation/causal claim, no cross-model pooled denominator')


def plot(data,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(4,2,figsize=(13,13),constrained_layout=True)
    colors={'smoothquant-w4a8':'#e88922','smoothquant-w4a4':'#3274a1'}
    suites=['libero_spatial','libero_object','libero_goal','libero_10']
    for index,suite in enumerate(suites):
        rows=[r for r in data['pairs'] if r['suite']==suite]
        for col,(key,label,threshold) in enumerate([
            ('translation_error_mm','Translation deviation (mm)','translation_mm'),
            ('rotation_geodesic_deg','SO(3) deviation (deg)','rotation_deg')]):
            ax=axes[index,col]
            for row in rows:
                ax.plot(row['physical_control_step'],row[key],label=row['condition'],color=colors[row['condition']])
            if data['thresholds']:ax.axhline(data['thresholds'][threshold],color='#888',ls='--',lw=1,label='Repeat-calibrated threshold')
            ax.set_title(suite+' | task 0 / initial 5');ax.set_ylabel(label);ax.set_xlabel('Physical control step')
            ax.grid(alpha=.18)
            if not rows:ax.text(.5,.5,'No completed paired trajectory',ha='center',transform=ax.transAxes)
    if data['pairs']:axes[0,0].legend(fontsize=8)
    fig.suptitle('GR00T-N1.5: same-backend closed-loop drift (pilot coverage)',fontsize=15)
    fig.savefig(output/'groot-physical-step-drift.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for method,color in colors.items():
        rows=[r for r in data['pairs'] if r['condition']==method]
        for ax,key,label in [(axes[0],'translation_error_mm','Translation deviation (mm)'),(axes[1],'rotation_geodesic_deg','SO(3) deviation (deg)')]:
            if not rows:continue
            maximum=max(len(r[key]) for r in rows);med=[];lower=[];upper=[];at_risk=[]
            for i in range(maximum):
                values=[r[key][i] for r in rows if i<len(r[key])]
                med.append(np.median(values));lower.append(np.percentile(values,25));upper.append(np.percentile(values,75));at_risk.append(len(values))
            x=np.arange(1,maximum+1);ax.plot(x,med,color=color,label=method+' (descriptive median)')
            ax.fill_between(x,lower,upper,color=color,alpha=.14);ax.set_ylabel(label);ax.set_xlabel('Physical control step')
            data.setdefault('at_risk',{}).setdefault(method,{})[key]=at_risk
            if key=='translation_error_mm':
                axes[2].step(x,at_risk,where='post',color=color,label=method)
    axes[2].set_xlabel('Physical control step');axes[2].set_ylabel('Available paired trajectories')
    axes[2].set_yticks([0,1,2,3,4]);axes[2].set_ylim(0,4.3)
    for ax in axes:ax.grid(alpha=.18);ax.legend(fontsize=8)
    fig.suptitle('Four pilot tasks: median and IQR; available prefixes only (n <= 4)')
    fig.savefig(output/'groot-pilot-aggregate.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--round-dir',type=Path);p.add_argument('--data',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--plot',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    data=json.loads(a.data.read_text()) if a.data else summarize(a.round_dir)
    if a.plot:plot(data,a.output)
    (a.output/'groot-drift-summary.json').write_text(json.dumps(data,indent=2)+'\n')
