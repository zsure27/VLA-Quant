"""Cross-backbone, cross-quantizer runtime-error commonality views.

--demo constructs scalar distributions, NOT VLA rollouts or paper measurements.
--records requires a private, content-bound analysis admission. No model loader.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from qvla.runtime_error_protocol_v3 import complete_case_ids,reflected_kde,scalar_summary

MODELS=['pi05','groot_n15']
PRECISIONS=['BF16','ActQuant-W4A16','SQ-W4A8','QuantVLA-W4A8']
SUITES=['libero_spatial','libero_object','libero_goal','libero_10']
TIMES=list(range(13))


def demo():
    rows=[]
    for mi,model in enumerate(MODELS):
        for si,suite in enumerate(SUITES):
            for task in (0,5):
                for reset in (5,6,7):
                    for di,direction in enumerate(('Tz+','Tz-','Rz+','Rz-')):
                        rng=np.random.RandomState(10000*mi+1000*si+100*task+10*reset+di)
                        latent=rng.normal(0,1,(len(TIMES),4))
                        for pi,precision in enumerate(PRECISIONS):
                            spread=[1,2.4,1.7,1.3][pi];response=[1,1.8,1.35,1.1][pi]
                            for tau in TIMES:
                                # Planned missing support illustrates censoring; no success selection.
                                if precision=='ActQuant-W4A16' and task==5 and reset==7 and tau>=9:continue
                                epsT=max(.003,.12+.015*si+.045*spread*latent[tau,0])
                                epsR=max(.003,.025+.012*tau+.004*latent[tau,1])*(1+.025*pi)
                                dT=max(.001,epsT*response*(.65+.12*abs(latent[tau,2])))
                                dR=max(.001,epsR*(.5+.04*pi+.06*abs(latent[tau,3])))
                                all_delta=float(np.hypot(dT,dR))
                                rows.append(dict(model=model,suite=suite,task_id=task,reset_index=reset,
                                    case_id=f'{suite}-task{task}-reset{reset}-{direction}',precision=precision,
                                    direction=direction,tau=tau,elapsed_control_steps=tau*(5 if model=='pi05' else 1),
                                    epsilon_T=epsT,epsilon_R=epsR,delta_A_T_next=dT,delta_A_R_next=dR,
                                    sensitivity_T=all_delta/epsT,sensitivity_R=all_delta/epsR,
                                    protocol='DYQ_SECTION3_PAIRED_V3'))
    return dict(data_kind='synthetic_demo',note='Constructed scalar populations; NOT benchmark evidence',
                models=MODELS,precisions=PRECISIONS,time_grid=TIMES,records=rows)


def cohort(bundle,model,direction):
    rows=[r for r in bundle['records'] if r['model']==model and r['direction']==direction]
    complete,risk=complete_case_ids(rows,bundle['precisions'],bundle['time_grid'])
    return [r for r in rows if r['case_id'] in complete],risk,len(complete)


def stats(bundle):
    result={}
    for model in bundle['models']:
        result[model]={}
        for component,direction in bundle.get('display_directions',{'T':'Tz+','R':'Rz+'}).items():
            rows,risk,n=cohort(bundle,model,direction)
            if n<2:raise ValueError('At least two complete paired rollouts required for dispersion views')
            panels={}
            for precision in bundle['precisions']:
                panels[precision]={}
                for t in bundle['time_grid']:
                    r=[x for x in rows if x['precision']==precision and x['tau']==t]
                    panels[precision][str(t)]=scalar_summary([x['epsilon_'+component] for x in r])
            # Available-case companion includes early terminated/missing-later cases at earlier times.
            available={}
            for precision in bundle['precisions']:
                available[precision]={}
                for t in bundle['time_grid']:
                    values=[r['epsilon_'+component] for r in bundle['records'] if r['model']==model and r['direction']==direction and r['precision']==precision and r['tau']==t]
                    available[precision][str(t)]=scalar_summary(values) if values else None
            result[model][component]=dict(complete_n=n,risk_set=risk,complete_case=panels,available_case=available)
    return result


def figures(bundle,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models,precisions,times=bundle['models'],bundle['precisions'],bundle['time_grid']
    directions=bundle.get('display_directions',{'T':'Tz+','R':'Rz+'})
    colors={p:plt.cm.tab10(i) for i,p in enumerate(precisions)}
    title='SYNTHETIC DEMO - NOT VLA EXPERIMENT RESULTS' if bundle['data_kind']=='synthetic_demo' else 'MEASURED | SAME-PRECISION PERTURBED PAIRS'
    files=[];summary=stats(bundle)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.suptitle(title,fontsize=13);fig.tight_layout(rect=(0,0,1,.92))
        for ext in ('png','pdf'):fig.savefig(str(out/(name+'.'+ext)),dpi=145)
        plt.close(fig);files.append(name+'.png')
    # Fixed axes/bandwidth at all checkpoints; example quantizer not selected by effect.
    comparison='SQ-W4A8';shown=[times[0],times[len(times)//2],times[-1]]
    fig,axes=plt.subplots(4,3,figsize=(12,14))
    for mi,model in enumerate(models):
        for ci,(component,direction) in enumerate(directions.items()):
            rows,risk,n=cohort(bundle,model,direction)
            selected=[r for r in rows if r['precision'] in ('BF16',comparison)]
            pooled=np.array([[r['delta_A_'+component+'_next'],r['epsilon_'+component]] for r in selected])
            bandwidth=np.maximum(pooled.std(axis=0)*n**(-1/6),np.maximum(pooled.max(axis=0)*.02,1e-4))
            xmax,ymax=pooled.max(axis=0)+3*bandwidth
            gx=np.linspace(0,xmax,65);gy=np.linspace(0,ymax,65)
            for ti,t in enumerate(shown):
                ax=axes[mi*2+ci,ti]
                for precision,style in [('BF16','--'),(comparison,'-')]:
                    a=np.array([[r['delta_A_'+component+'_next'],r['epsilon_'+component]] for r in selected if r['precision']==precision and r['tau']==t])
                    z=reflected_kde(a,gx,gy,bandwidth)
                    levels=[z.max()*.25,z.max()*.6]
                    ax.contour(gx,gy,z,levels=levels,colors=[colors[precision]],linestyles=style,linewidths=1.1)
                    mu=a.mean(axis=0);ax.plot(mu[0],mu[1],'o' if precision=='BF16' else 's',color=colors[precision],label=precision)
                ax.set(xlim=(0,xmax),ylim=(0,ymax),xlabel='Next-query action response (scaled)',ylabel='Current '+component+' pose error (scaled)',
                       title=model+' / '+component+' / query +'+str(t)+' / fixed N='+str(n))
                if ti==0:ax.legend(fontsize=7)
    save(fig,'14-joint-distribution-evolution')
    for field,filename,ylabel in [('mean','15-distribution-centers','Mean normalized pose error'),('std','16-distribution-dispersion','SD across paired rollouts')]:
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for mi,model in enumerate(models):
            for ci,component in enumerate(('T','R')):
                ax=axes[mi,ci];s=summary[model][component]
                for precision in precisions:
                    ax.plot(times,[s['complete_case'][precision][str(t)][field] for t in times],label=precision,color=colors[precision],ls='--' if precision=='BF16' else '-')
                ax.set(title=model+' / '+component+' / fixed N='+str(s['complete_n']),xlabel='Queries after perturbation (x'+str(5 if model=='pi05' else 1)+' control steps)',ylabel=ylabel)
                ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8)
        save(fig,filename)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for mi,model in enumerate(models):
        for ci,(component,direction) in enumerate(directions.items()):
            rows,risk,n=cohort(bundle,model,direction);ax=axes[mi,ci]
            for precision in precisions:
                values=[]
                for t in times:
                    v=[r['sensitivity_'+component] for r in rows if r['precision']==precision and r['tau']==t and r['sensitivity_'+component] is not None]
                    values.append(float(np.median(v)) if v else np.nan)
                ax.plot(times,values,label=precision,color=colors[precision],ls='--' if precision=='BF16' else '-')
            ax.set(title=model+' / response per unit '+component+' error',xlabel='Queries after perturbation (x'+str(5 if model=='pi05' else 1)+' control steps)',ylabel='Median sensitivity (all 6 action channels)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);save(fig,'17-next-query-response-sensitivity')
    # A consistency matrix is evidence by suite/model, not a made-up overall score.
    fig,axes=plt.subplots(2,3,figsize=(14,8));matrix_values={};matrix_images={0:[],1:[],2:[]}
    for mi,model in enumerate(models):
        for metric in range(3):
            grid=np.full((4,len(precisions)-1),np.nan)
            for si,suite in enumerate(SUITES):
                for pi,precision in enumerate(precisions[1:]):
                    comp='R' if metric==1 else 'T';direction=directions[comp]
                    rows,risk,n=cohort(bundle,model,direction)
                    subset=[r for r in rows if r['suite']==suite]
                    end=[r for r in subset if r['tau']==times[-1] and r['precision']==precision]
                    fp=[r for r in subset if r['tau']==times[-1] and r['precision']=='BF16']
                    start=[r for r in subset if r['tau']==times[0] and r['precision']==precision]
                    if len(end)<2:continue
                    if metric==0:
                        denominator=np.var([r['epsilon_T'] for r in fp]);value=np.var([r['epsilon_T'] for r in end])/denominator if denominator>1e-12 else np.nan
                    elif metric==1:
                        fp_start=[r for r in subset if r['tau']==times[0] and r['precision']=='BF16']
                        value=(np.mean([r['epsilon_R'] for r in end])-np.mean([r['epsilon_R'] for r in start]))-(np.mean([r['epsilon_R'] for r in fp])-np.mean([r['epsilon_R'] for r in fp_start]))
                    else:
                        a={r['case_id']:r['sensitivity_T'] for r in end};b={r['case_id']:r['sensitivity_T'] for r in fp}
                        ratios=[a[k]/b[k] for k in a if a[k] is not None and b[k] is not None and b[k]>1e-8]
                        value=float(np.median(ratios)) if ratios else np.nan
                    grid[si,pi]=value
            im=axes[mi,metric].imshow(np.ma.masked_invalid(grid),aspect='auto',cmap='viridis')
            axes[mi,metric].set_yticks(range(4));axes[mi,metric].set_yticklabels(['Spatial','Object','Goal','Long'])
            axes[mi,metric].set_xticks(range(len(precisions)-1));axes[mi,metric].set_xticklabels(precisions[1:],rotation=25,ha='right',fontsize=8)
            axes[mi,metric].set_title(model+' / '+['T variance ratio to FP','Extra R center change vs FP','T sensitivity ratio to FP'][metric])
            for si in range(4):
                for pi in range(len(precisions)-1):
                    if np.isfinite(grid[si,pi]):axes[mi,metric].text(pi,si,format(grid[si,pi],'.3f' if metric==1 else '.2f'),ha='center',va='center',color='white',fontsize=9)
            fig.colorbar(im,ax=axes[mi,metric]);matrix_images[metric].append((im,grid))
            matrix_values[model+'-'+str(metric)]=[[float(v) if np.isfinite(v) else None for v in row] for row in grid]
    for metric,images in matrix_images.items():
        values=np.concatenate([g[np.isfinite(g)] for im,g in images])
        if len(values):
            lo=min(0.,float(values.min()));hi=max(float(values.max()),1e-8)
            for im,g in images:
                im.set_clim(lo,hi)
                for annotation in im.axes.texts:
                    x,y=annotation.get_position()
                    annotation.set_color('black' if im.norm(g[int(y),int(x)])>.6 else 'white')
    save(fig,'18-suite-commonality-evidence')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,model in zip(axes,models):
        s=summary[model]['T']
        for precision in precisions:ax.step(times,[s['risk_set'][precision][str(t)] for t in times],where='mid',label=precision,color=colors[precision])
        ax.axhline(s['complete_n'],color='gray',ls=':',label='Fixed complete cohort')
        ax.set(title=model+' / '+directions['T']+' support',xlabel='Queries after perturbation',ylabel='Paired rollouts with valid next-query record',ylim=(0,max(max(v.values()) for v in s['risk_set'].values())+2));ax.legend(fontsize=7)
    save(fig,'19-risk-set-and-complete-cohort')
    summary['matrix_values']=matrix_values
    return files,summary


def main():
    p=argparse.ArgumentParser(description=__doc__);group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--demo',action='store_true');group.add_argument('--records',type=Path)
    p.add_argument('--translation-direction',choices=['Tz+','Tz-'],default='Tz+')
    p.add_argument('--rotation-direction',choices=['Rz+','Rz-'],default='Rz+')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Use a fresh analysis output directory')
    bundle=demo() if args.demo else json.loads(args.records.read_text(encoding='utf-8'))
    if not args.demo:
        if bundle.get('data_kind')!='measured':raise ValueError('Measured record bundle required')
        gate_path=args.records.parent/bundle['admission_file'];gate=json.loads(gate_path.read_text(encoding='utf-8'))
        content={k:v for k,v in bundle.items() if k not in ('admission_file','admission_sha256')}
        digest=hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if hashlib.sha256(gate_path.read_bytes()).hexdigest()!=bundle['admission_sha256'] or gate.get('content_sha256')!=digest or gate.get('gate')!='PASS_DYQ_SECTION3_PAIRED_DATA_V3':raise ValueError('Unverified measured cohort')
    for r in bundle['records']:
        if r.get('protocol')!='DYQ_SECTION3_PAIRED_V3' or r['model'] not in MODELS:raise ValueError('Wrong reference/model protocol')
        for field in ('epsilon_T','epsilon_R','delta_A_T_next','delta_A_R_next'):
            if not np.isfinite(r[field]) or r[field]<0:raise ValueError('Invalid nonnegative finite metric: '+field)
        for field in ('sensitivity_T','sensitivity_R'):
            if r[field] is not None and (not np.isfinite(r[field]) or r[field]<0):raise ValueError('Invalid sensitivity: '+field)
    if bundle['models']!=MODELS or bundle['precisions']!=PRECISIONS:raise ValueError('This version requires the preregistered two models and four conditions')
    bundle['display_directions']={'T':args.translation_direction,'R':args.rotation_direction}
    args.output.mkdir(parents=True);files,summary=figures(bundle,args.output)
    (args.output/'summary.json').write_text(json.dumps({'data_kind':bundle['data_kind'],'display_directions':bundle['display_directions'],'time_grid':bundle['time_grid'],'summary':summary},allow_nan=False,indent=2)+'\n',encoding='utf-8')
    label='合成分布示意，不是模型运行或论文实测数据' if args.demo else '实测配对分支诊断'
    html='<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><style>body{font:16px system-ui;max-width:1250px;margin:auto;padding:16px}header{background:#fff1c9;padding:16px;position:sticky;top:0}img{width:100%;height:auto}figure{margin:30px 0}</style><header>'+label+'</header><h1>跨模型、跨方法的运行时误差共性</h1><p>实线/方形为量化，虚线/圆形为BF16。每组均比较同精度的扰动分支与无扰动分支。两模型的原生query跨度不同，按模型分面，不混合计算。固定完整cohort与可用样本数同时报告。</p>'
    captions=['同一坐标范围和带宽比较早中晚联合分布；点为原始样本均值，等高线为密度峰值的25%/60%，不是置信区间。',
              '中心是否逐步移动？位置差范数和姿态差角分别展示，不能由范数推断有符号漂移方向。',
              '不同配对轨迹之间的离散度是否放大？这不是一条轨迹内滑窗波动。',
              '当前位姿误差对应下一次查询的六维动作响应；近零误差不计算比值。',
              '各suite和模型中的效应是否同向？旋转格为量化的中心变化减去BF16中心变化；缺失不补零。',
              '同时保留逐查询风险集与跨方法/时间的固定完整样本集，避免后期样本选择被隐藏。']
    html+='<p>当前展示方向：'+args.translation_direction+' / '+args.rotation_direction+'；负向扰动必须另生成对照，不能合并抵消。</p>'
    html+=''.join('<figure><img src="'+f+'" alt="'+f+'"><figcaption>'+caption+'</figcaption></figure>' for f,caption in zip(files,captions))
    (args.output/'index.html').write_text(html,encoding='utf-8')
    print(json.dumps(dict(data_kind=bundle['data_kind'],figures=len(files),measured_rollouts=0 if args.demo else 'see admitted manifest')))


if __name__=='__main__':main()
