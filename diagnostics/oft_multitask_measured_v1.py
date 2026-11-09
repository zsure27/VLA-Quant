"""Measured four-suite OFT analysis, preserving missing cases and frozen pilot rules."""
from pathlib import Path
import json, hashlib
import numpy as np
from diagnostics.native_oft_trace_v1 import load, compare, AUDITS, SUITES, CONDITIONS
from qvla.trajectory_diagnostics_v1 import sustained_windows

ROOT = Path('runs')
OUT = Path('analysis/20261109-oft-multitask-v1')
def reference(s, task):
    if task:
        return ROOT/f'20261109-053-bf16-{s}-task{task}-reset5-v1'
    return ROOT/('20261108-053-oft-bf16-admission-v3' if s == 'libero_spatial' else f'20261108-053-oft-bf16-{s}-v1')
def main():
    OUT.mkdir(parents=True, exist_ok=False)
    threshold_path = Path('analysis/20261109-oft-pilot-v1/frozen-thresholds.json')
    threshold_bytes = threshold_path.read_bytes()
    thresholds = json.loads(threshold_bytes)
    (OUT/'frozen-thresholds.json').write_bytes(threshold_bytes)
    records, missing, traces = [], [], {}
    for s in SUITES:
        for task in range(10):
            ref = reference(s,task)
            for c in CONDITIONS:
                cand = ROOT/f'20261109-053-{c}-{s}-task{task}-reset5-v1'
                if not (ref/'result.json').exists() or not (cand/'result.json').exists():
                    missing.append(dict(suite=s,task_id=task,condition=c,reference=ref.name,candidate=cand.name,reason='missing terminal result; not filled as failure or zero error'))
                    continue
                a,b = load(cand),load(ref)
                t,r,w = compare(a,b)
                delta = a['first_chunk'].astype(float)-b['first_chunk'].astype(float)
                executed = min(8,len(a['steps']),len(b['steps']))
                row=dict(suite=s,task_id=task,reset_index=5,condition=c,success=a['result']['success'],reference_success=b['result']['success'],
                    candidate_steps=len(a['steps']),reference_steps=len(b['steps']),common_steps=len(t),
                    translation_p95_mm=float(np.percentile(t,95)*1000),rotation_p95_deg=float(np.percentile(r,95)),
                    translation_windows=sustained_windows(t,thresholds['translation_m'],thresholds['persistence']),
                    rotation_windows=sustained_windows(r,thresholds['rotation_deg'],thresholds['persistence']),
                    first_chunk_same_input_sha256=a['first_input_sha'],first_chunk_executed_positions=executed,
                    first_chunk_translation_command_rmse=float(np.sqrt(np.mean(delta[:executed,:3]**2))),
                    first_chunk_rotation_command_rmse=float(np.sqrt(np.mean(delta[:executed,3:6]**2))),
                    first_chunk_gripper_decision_mismatches=int(sum(a['steps'][i]['controller_command'][-1] != b['steps'][i]['controller_command'][-1] for i in range(executed))),
                    wall_s_including_load=a['result']['wall_s_including_load'])
                records.append(row);traces[s,task,c]=(t,r,w,delta,executed)
    summary=dict(scope='OFT, four suites, tasks0-9, initial index5 only; development tasks; incomplete cells explicitly retained',
        planned_pairs=120,available_pairs=len(records),missing=missing,thresholds=thresholds,
        threshold_sha256=hashlib.sha256(threshold_bytes).hexdigest(),metrics=records,
        interpretation='Matched-reset pose differences include feedback state divergence. Only first-query action contrasts share exact inputs. No OPD intervention or five-initial-state claim.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    (OUT/'raw-trace-audit.json').write_text(json.dumps(dict(gate='PASS_AVAILABLE_MULTITASK_RAW_TRACE_AUDIT',episodes=AUDITS,missing=missing),indent=2))
    figures(records,traces,thresholds)
    print(json.dumps(dict(available_pairs=len(records),missing_pairs=len(missing),audited_episodes=len(AUDITS))))

def figures(records,traces,thresholds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#2878b5','#e07b39','#b44770'];files=[]
    def save(fig,name):
        fig.suptitle('MEASURED OFT | 4 suites / up to 10 tasks each / initial index 5',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.96))
        for ext in ('png','pdf'):fig.savefig(OUT/f'{name}.{ext}',dpi=150)
        plt.close(fig);files.append(name+'.png')
    fig,axes=plt.subplots(3,4,figsize=(17,10))
    for j,s in enumerate(SUITES):
        for c,color in zip(CONDITIONS,colors):
            rows=[traces[s,t,c] for t in range(10) if (s,t,c) in traces]
            if not rows:continue
            for metric,factor in ((0,1000),(1,1)):
                width=max(len(x[metric]) for x in rows);grid=np.full((len(rows),width),np.nan)
                for i,x in enumerate(rows):grid[i,:len(x[metric])]=x[metric]*factor
                count=np.isfinite(grid).sum(axis=0);med=np.nanmedian(grid,axis=0);lo,hi=np.nanpercentile(grid,[25,75],axis=0)
                axes[metric,j].plot(med,color=color,label=c);axes[metric,j].fill_between(np.arange(width),lo,hi,color=color,alpha=.15)
                if metric==0:axes[2,j].step(np.arange(width),count,color=color,where='post',label=c)
        axes[0,j].axhline(thresholds['translation_m']*1000,color='gray',ls='--');axes[1,j].axhline(thresholds['rotation_deg'],color='gray',ls='--')
        axes[0,j].set_title(s);axes[0,j].legend(fontsize=7);axes[0,j].set_ylabel('Median / IQR position diff (mm)');axes[1,j].set_ylabel('Median / IQR SO(3) diff (deg)');axes[2,j].set_ylabel('Observed paired tasks');axes[2,j].set_xlabel('Raw control step; no time warping')
    save(fig,'01-control-step-drift-and-risk-set')
    fig,axes=plt.subplots(1,3,figsize=(15,12))
    keys=['translation_p95_mm','rotation_p95_deg','first_chunk_translation_command_rmse']
    for ax,key in zip(axes,keys):
        grid=np.full((40,3),np.nan)
        for row in records:grid[SUITES.index(row['suite'])*10+row['task_id'],CONDITIONS.index(row['condition'])]=row[key]
        im=ax.imshow(grid,aspect='auto');ax.set_xticks(range(3));ax.set_xticklabels(CONDITIONS,rotation=45,ha='right',fontsize=8);ax.set_yticks(range(40));ax.set_yticklabels([s.replace('libero_','')+'/'+str(t) for s in SUITES for t in range(10)],fontsize=7);ax.set_title(key,fontsize=9);fig.colorbar(im,ax=ax)
    save(fig,'02-task-method-error-map')
    fig,axes=plt.subplots(2,4,figsize=(17,8))
    for j,s in enumerate(SUITES):
        for c,color in zip(CONDITIONS,colors):
            rows=[traces[s,t,c] for t in range(10) if (s,t,c) in traces]
            if not rows:continue
            for k,sl in ((0,slice(0,3)),(1,slice(3,6))):
                grid=np.full((len(rows),8),np.nan)
                for i,x in enumerate(rows):grid[i,:x[4]]=np.sqrt(np.mean(x[3][:x[4],sl]**2,axis=1))
                axes[k,j].plot(range(8),np.nanmedian(grid,axis=0),color=color,marker='o',label=f'{c}, tasks={len(rows)}')
            axes[0,j].set_title(s);axes[0,j].legend(fontsize=6)
        axes[0,j].set_ylabel('Translation command RMSE');axes[1,j].set_ylabel('Rotation command RMSE');axes[1,j].set_xlabel('Executed position in first 8-action chunk')
    save(fig,'03-exact-input-initial-action-errors')
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    for ax,key,label in zip(axes,['translation_windows','rotation_windows'],['First persistent position deviation','First persistent rotation deviation']):
        for ci,c in enumerate(CONDITIONS):
            ys=[x[key][0]['onset_index'] for x in records if x['condition']==c and x[key]]
            if ys:ax.scatter(np.full(len(ys),ci)+np.linspace(-.12,.12,len(ys)),ys,color=colors[ci],s=20,alpha=.6)
            available=sum(x['condition']==c for x in records);ax.text(ci,1.01,f'windows {len(ys)}/{available}',ha='center',transform=ax.get_xaxis_transform(),fontsize=8)
        ax.set_xticks(range(3));ax.set_xticklabels(CONDITIONS,rotation=25,ha='right');ax.set_ylabel('Onset control step; confirmed 4 steps later');ax.set_title(label,pad=25)
    save(fig,'04-persistent-window-task-variation')
    (OUT/'index.html').write_text('<h1>Measured multi-task OFT drift</h1><p>One initial state per task. Missing cases remain missing. Pose differences describe closed-loop divergence; the initial-action figure uses exactly matched input SHA. Windows use predeclared BF16-repeat thresholds and are not mathematical curvature points.</p>'+''.join(f'<h2>{p}</h2><img style="width:100%" src="{p}">' for p in files))

if __name__=='__main__':main()
