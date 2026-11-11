"""Reuse frozen OFT raw-audit/metric math; extend only clone paths and profile lookup."""
import argparse
import json
from pathlib import Path
import sys
import types

import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--round-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root,rd,out=a.root.resolve(),a.round_dir.resolve(),a.output.resolve()
    if out.exists():raise FileExistsError('Use a new analysis version; preserve old evidence')
    sys.path[:0]=[str(root/'code-cpu-prep-20261109-v1'),str(root/'sources/openvla-oft'),str(root/'sources/LIBERO')]
    from qvla.infra.common import digest
    from qvla.trajectory_diagnostics_v1 import sustained_windows
    series=json.loads((rd/'series.json').read_text());profiles={key:root/value['path'] for key,value in series['profiles'].items()}
    source=root/'code-cpu-prep-20261109-v1/diagnostics/native_oft_trace_v1.py';body=source.read_text()
    old="profile = 'awq.pt' if m['condition'].startswith('awq') else 'smoothquant.pt'\n        assert result['profile_sha256'] == sha(Path('profiles/20261109-four-suite-reset0-v4')/profile)"
    if body.count(old)!=1:raise ValueError('Frozen raw auditor differs')
    body=body.replace(old,"assert result['profile_sha256'] == sha(profile_paths[m['condition']])")
    module=types.ModuleType('frozen_oft_clone_audit');module.profile_paths=profiles
    exec(compile(body,str(source)+'[profile path binding only]','exec'),module.__dict__)
    hashes={}
    def cached_sha(file):
        path=Path(file);st=path.stat();key=(str(path),st.st_size,st.st_mtime_ns)
        if key not in hashes:hashes[key]=digest(path)
        return hashes[key]
    module.sha=cached_sha;lookup={};origins={}
    for row in series['inherited_completed']:
        if row['reset_index']!=15:continue
        key=row['suite'],row['task_id'],row['condition'];path=Path(row['result_path'])
        if digest(path)!=row['result_sha256']:raise ValueError('Inherited result SHA changed')
        if key in lookup:raise ValueError('Duplicate inherited key')
        lookup[key]=path.parent;origins[key]='053'
    for status in rd.glob('libero*-cases.json'):
        for row in json.loads(status.read_text())['completed']:
            case=row['case'];key=case['suite'],case['task_id'],case['condition'];path=Path(row['result_path'])
            if key in lookup:raise ValueError('Clone repeated a completed scientific key')
            if digest(path)!=row['result_sha256']:raise ValueError('New result SHA changed')
            lookup[key]=path.parent;origins[key]='061'
    threshold_file=root/'analysis/20261109-oft-pilot-v1/frozen-thresholds.json';thresholds=json.loads(threshold_file.read_text())
    conditions=['awq_w4a16','gptq_w4a16','omniquant_w4a16','smoothquant_w4a8','smoothquant_w4a4']
    pairs={};metrics=[];missing=[];baselines={}
    for suite in module.SUITES:
        for task in range(10):
            reference=lookup.get((suite,task,'bf16'))
            if reference:baselines[suite,task]=module.load(reference)
            for condition in conditions:
                key=suite,task,condition
                if not reference or key not in lookup:
                    missing.append(dict(suite=suite,task_id=task,condition=condition));continue
                cand,ref=module.load(lookup[key]),baselines[suite,task]
                t,rot,w=module.compare(cand,ref);pairs[key]=(cand,ref,t,rot,w)
                tw=sustained_windows(t,thresholds['translation_m'],thresholds['persistence'])
                rw=sustained_windows(rot,thresholds['rotation_deg'],thresholds['persistence'])
                metrics.append(dict(suite=suite,task_id=task,condition=condition,reset_index=15,origin=origins[key],
                    success=cand['result']['success'],reference_success=ref['result']['success'],common_control_steps=len(t),
                    translation_p95_mm=float(np.percentile(t,95)*1000),rotation_p95_deg=float(np.percentile(rot,95)),
                    translation_windows=tw,rotation_windows=rw,first_input_sha256=cand['first_input_sha'],
                    first_same_input_chunk_mse=float(np.mean((cand['first_chunk']-ref['first_chunk'])**2)),
                    wall_seconds_including_load=cand['result']['wall_s_including_load']))
    out.mkdir(parents=True)
    summary=dict(gate='PASS_OFT_CLONE_AVAILABLE_RAW_QUERY_ACTION_POSE_VIDEO_AND_PAIRED_AUDIT',reset_index=15,
                 planned_pairs=200,available_pairs=len(metrics),missing=missing,metrics=metrics,
                 baseline_completed=len(baselines),baseline_success=sum(bool(x['result']['success']) for x in baselines.values()),
                 threshold_sha256=digest(threshold_file),frozen_raw_auditor_sha256=digest(source),
                 scope='Same frozen OFT recipe across 053->061; one initial state15; no five-initial or cross-model claim',
                 thresholds=thresholds,per_condition={c:dict(complete=sum(x['condition']==c for x in metrics),success=sum(x['condition']==c and x['success'] for x in metrics)) for c in conditions})
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'raw-audit.json').write_text(json.dumps(module.AUDITS,indent=2)+'\n')
    plots(out,pairs,baselines,module.SUITES,conditions)
    print(json.dumps(dict(gate=summary['gate'],paired=len(metrics),missing=len(missing),output=str(out))))


def plots(out,pairs,baselines,suites,conditions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(zip(conditions,['#2878b5','#268c70','#7956a3','#e07b39','#b44770']))
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=145,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(4,3,figsize=(17,13),constrained_layout=True);risk=[]
    for i,suite in enumerate(suites):
        for condition in conditions:
            rows=[pairs[suite,t,condition] for t in range(10) if (suite,t,condition) in pairs]
            if not rows:continue
            for col,slot,factor,label in [(0,2,1000,'Translation deviation (mm)'),(1,3,1,'SO(3) deviation (deg)')]:
                width=max(len(x[slot]) for x in rows);grid=np.full((len(rows),width),np.nan)
                for j,row in enumerate(rows):grid[j,:len(row[slot])]=row[slot]*factor
                count=np.isfinite(grid).sum(0);median=np.nanmedian(grid,0);x=np.arange(1,width+1)
                axes[i,col].plot(x,median,color=colors[condition],label=condition)
                axes[i,col].fill_between(x,np.nanpercentile(grid,25,axis=0),np.nanpercentile(grid,75,axis=0),color=colors[condition],alpha=.1)
                axes[i,col].set_ylabel(label)
                if col==0:
                    axes[i,2].step(x,count,where='post',color=colors[condition]);risk.append(dict(suite=suite,condition=condition,counts=count.tolist()))
        for ax in axes[i]:ax.set_xlabel('Physical control step');ax.set_title(suite+' / initial15');ax.grid(alpha=.15)
        axes[i,2].set_ylabel('Available paired trajectories');axes[i,2].set_ylim(0,10.5)
    axes[0,0].legend(fontsize=7);fig.suptitle('Same-recipe OFT drift across cloned servers: median/IQR and risk sets')
    save(fig,'01-physical-step-drift-and-risk-set');(out/'risk-sets.json').write_text(json.dumps(risk)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    labels=[];trans=[];rot=[]
    for condition in conditions:
        rows=[v for k,v in pairs.items() if k[2]==condition]
        if not rows:continue
        labels.append(condition);trans.append([np.percentile(x[2],95)*1000 for x in rows]);rot.append([np.percentile(x[3],95) for x in rows])
    if labels:
        for ax,values,title in [(axes[0],trans,'Per-task translation P95 (mm)'),(axes[1],rot,'Per-task SO(3) P95 (deg)')]:
            ax.boxplot(values,showmeans=True);ax.set_xticks(range(1,len(labels)+1),labels);ax.tick_params(axis='x',rotation=30);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    save(fig,'02-task-error-distributions')
    fig=plt.figure(figsize=(18,5))
    for i,suite in enumerate(suites):
        ax=fig.add_subplot(1,4,i+1,projection='3d');key=suite,0
        if key not in baselines:ax.set_title(suite+' / missing BF16');continue
        xyz=baselines[key]['position']*1000;ax.plot(*xyz.T,color='#444',label='BF16')
        for condition in conditions:
            if (suite,0,condition) not in pairs:continue
            cand,ref,t,_,_=pairs[suite,0,condition];xyz=cand['position'][:len(t)]*1000;ax.plot(*xyz.T,color=colors[condition],label=condition)
        ax.set_title(suite+' / task0 / initial15');ax.set_xlabel('Base X (mm)');ax.set_ylabel('Base Y (mm)');ax.set_zlabel('Base Z (mm)')
        if i==0:ax.legend(fontsize=6)
    save(fig,'03-prespecified-task0-3d-paths')
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    values=[[float(np.mean((v[0]['first_chunk']-v[1]['first_chunk'])**2)) for k,v in pairs.items() if k[2]==c] for c in labels]
    if labels:ax.boxplot(values,showmeans=True);ax.set_xticks(range(1,len(labels)+1),labels)
    ax.set_ylabel('Raw policy chunk MSE (native model units)');ax.set_title('Exactly matched first input: direct action discrepancy');ax.tick_params(axis='x',rotation=25)
    save(fig,'04-same-input-first-action-error')
    (out/'index.html').write_text('<meta charset="utf-8"><h1>OFT初态15：053→061相同配方接续</h1><p>实测共同控制步前缀；仅初态15；不补零、不混入未完成槽位，不证明补偿有效。</p>'+''.join('<h2>'+name+'</h2><img style="width:100%" src="'+name+'.png">' for name in ['01-physical-step-drift-and-risk-set','02-task-error-distributions','03-prespecified-task0-3d-paths','04-same-input-first-action-error']))


if __name__=='__main__':main()
