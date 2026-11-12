"""Audited available pairs, with explicit censoring and sample counts."""
import argparse,json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--audit-dirs',nargs='+',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
curves=[];metrics=[];keys=set();gates=set()
for d in a.audit_dirs:
 result=json.loads((d/'result.json').read_text());assert result['gate'] in ['PASS_QVLA_RAW_QUERY_ACTION_POSE_VIDEO_AND_PAIRED_CASES','PASS_QVLA_AVAILABLE_RAW_PAIRS_QUOTA_STOP']
 assert result['completed']==len(result['metrics']);gates.add(result['gates_sha256']);metrics+=result['metrics']
 for c in json.loads((d/'curves.json').read_text()):
  key=(c['suite'],c['task_id'],c['reset_index']);assert key not in keys;keys.add(key);curves.append(c)
assert len(gates)==1 and len(metrics)==len(curves)
suites=['libero_spatial','libero_object','libero_goal','libero_10'];fig,axes=plt.subplots(3,4,figsize=(15,8.5),squeeze=False,gridspec_kw={'height_ratios':[3,3,1]})
for col,suite in enumerate(suites):
 rows=[x for x in curves if x['suite']==suite];n=len(rows);m=[x for x in metrics if x['suite']==suite]
 for axis,(key,label,color) in enumerate([('translation_mm','Position separation (mm)','#0072B2'),('rotation_deg','Orientation separation (deg)','#D55E00')]):
  ax=axes[axis,col];ax.set_title(f'{suite} | {n} audited pairs',fontsize=10);ax.set_ylabel(label);ax.grid(alpha=.2)
  if not rows:ax.text(.5,.5,'No completed pair',transform=ax.transAxes,ha='center');continue
  length=max(len(x[key]) for x in rows);values=np.full((n,length),np.nan)
  for i,c in enumerate(rows):values[i,:len(c[key])]=c[key]
  count=np.isfinite(values).sum(axis=0);x=np.arange(length);median=np.nanmedian(values,axis=0);lo,hi=np.nanpercentile(values,[25,75],axis=0)
  supported=count>=3;ax.plot(x,np.where(supported,median,np.nan),color=color,lw=1.6);ax.plot(x,np.where(~supported,median,np.nan),color=color,lw=1,ls=':');ax.fill_between(x,lo,hi,where=supported,color=color,alpha=.2)
  if axis==1:
   axes[2,col].step(x,count,where='post',color='black',lw=1);axes[2,col].set_ylim(0,max(n+1,2));axes[2,col].set_ylabel('At risk n');axes[2,col].set_xlabel('Executed environment step')
 if m:axes[2,col].text(.98,.9,f"Success Q/BF16: {sum(x['success'] for x in m)}/{sum(x['reference_success'] for x in m)} of {n}",ha='right',va='top',transform=axes[2,col].transAxes,fontsize=8)
for col,suite in enumerate(suites):
 for axis,key in enumerate(['translation_mm','rotation_deg']):
  ceiling=max([max(c[key]) for c in curves if c['suite']==suite] or [1]);axes[axis,col].set_ylim(0,max(ceiling*1.05,0.001))
fig.suptitle(f'OpenVLA-OFT QVLA vs BF16 | {len(curves)} available audited pairs',fontsize=15)
fig.text(.5,.018,'Median + trajectory IQR (not confidence interval); no padding / time warp. Dotted: fewer than 3 pairs remain.\nFree-rollout separation includes observation divergence. Quota-limited coverage; no universal change-point claim.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.07,1,.96));a.output.mkdir(parents=True,exist_ok=True)
for ext in ['png','pdf']:fig.savefig(a.output/('qvla-available-risk-curves.'+ext),dpi=180)
summary=dict(completed=len(metrics),success=sum(x['success'] for x in metrics),reference_success=sum(x['reference_success'] for x in metrics),rescue=sum(x['success'] and not x['reference_success'] for x in metrics),break_count=sum(not x['success'] and x['reference_success'] for x in metrics),gates_sha256=next(iter(gates)),by_suite={s:dict(completed=sum(x['suite']==s for x in metrics),success=sum(x['suite']==s and x['success'] for x in metrics),reference_success=sum(x['suite']==s and x['reference_success'] for x in metrics)) for s in suites},case_keys=[list(x) for x in sorted(keys)],claim_scope='available paired development trajectories only; no universal changepoint, no full80 unless count80')
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
