"""Measured seven-condition figures, explicit paired risk sets and frozen window rules."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def make(out,metrics,traces,conditions,suites,thresholds,baseline_positions):
 colors=dict(zip(conditions,['#0072B2','#009E73','#CC79A7','#E69F00','#D55E00','#222222']))
 labels=dict(zip(conditions,['AWQ W4A16','GPTQ W4A16','Omni W4A16','SQ W4A8','SQ W4A4','QVLA mixed~W4A16']))
 weights=[conditions[i] for i in [0,1,2,5]];activations=conditions[3:5];files=[];coverage=[]
 def save(fig,name):
  fig.savefig(out/(name+'.png'),dpi=135);fig.savefig(out/(name+'.pdf'));plt.close(fig);files.append(name)
 def rows_for(s,c):return sorted([x for x in metrics if x['suite']==s and x['condition']==c],key=lambda x:(x['task_id'],x['reset_index']))
 def grid_for(rows,kind):
  values=[traces[x['suite'],x['task_id'],x['reset_index'],x['condition']][kind] for x in rows];width=max(map(len,values));tail=values[0].shape[1:];grid=np.full((len(rows),width,*tail),np.nan)
  for i,v in enumerate(values):grid[i,:len(v)]=v
  count=np.isfinite(grid[:,:,0] if tail else grid).sum(0);tasks=np.array([len({row['task_id'] for row,v in zip(rows,values) if len(v)>step}) for step in range(width)])
  return grid,count,tasks
 for group,title,tag in [(weights,'Weight quantization (BF16 activations)','weight'),(activations,'Weight + activation quantization','activation')]:
  fig,axes=plt.subplots(4,4,figsize=(17,11),gridspec_kw={'height_ratios':[3,3,1,1]},constrained_layout=True)
  for col,suite in enumerate(suites):
   for c in group:
    rows=rows_for(suite,c)
    for axis,kind in enumerate(['translation_mm','rotation_deg']):
     grid,count,tasks=grid_for(rows,kind);x=np.arange(grid.shape[1]);med=np.nanmedian(grid,0);lo,hi=np.nanpercentile(grid,[25,75],axis=0);valid=tasks>=5
     axes[axis,col].plot(x,np.where(valid,med,np.nan),color=colors[c],lw=1.4,label=labels[c]);axes[axis,col].fill_between(x,lo,hi,where=valid,color=colors[c],alpha=.1)
     if axis==0:
      axes[2,col].step(x,count,color=colors[c],lw=1,where='post');axes[3,col].step(x,tasks,color=colors[c],lw=1,where='post')
      coverage.append(dict(group=tag,suite=suite,condition=c,paired_counts=count.tolist(),distinct_task_counts=tasks.tolist(),draw_median_when_at_least_distinct_tasks=5))
   axes[0,col].set_title(suite+' |20 pairs /10 tasks');axes[0,col].axhline(5,color='gray',ls=':',lw=.8);axes[1,col].axhline(2,color='gray',ls=':',lw=.8)
   for axis in range(4):axes[axis,col].grid(alpha=.15);axes[axis,col].set_ylim(bottom=0)
   axes[0,col].set_ylabel('Position separation (mm)');axes[1,col].set_ylabel('SO(3) separation (deg)');axes[2,col].set_ylabel('Pairs at risk');axes[2,col].set_ylim(0,21);axes[3,col].set_ylabel('Tasks at risk');axes[3,col].set_ylim(0,11);axes[3,col].set_xlabel('Executed environment step')
  axes[0,0].legend(fontsize=7);fig.suptitle(title+' | matched initial5/15, raw common prefixes\nMedian + trajectory IQR; at least5 distinct tasks remain. Grey: frozen5mm/2deg, not safety limits.');save(fig,'01-'+tag+'-drift-risk')
  fig,axes=plt.subplots(2,4,figsize=(17,8),constrained_layout=True)
  for col,suite in enumerate(suites):
   for c in group:
    rows=rows_for(suite,c)
    for axis,kind in enumerate(['translation_vector_mm','rotation_vector_deg']):
     grid,count,tasks=grid_for(rows,kind);mean=np.nanmean(grid,0);bias=np.linalg.norm(mean,axis=1);scatter=np.sqrt(np.nanmean(np.sum((grid-mean)**2,axis=2),0));valid=tasks>=5
     axes[axis,col].plot(np.where(valid,bias,np.nan),color=colors[c],label=labels[c]);axes[axis,col].plot(np.where(valid,scatter,np.nan),color=colors[c],ls='--',lw=1)
   for axis in range(2):axes[axis,col].grid(alpha=.15);axes[axis,col].set_ylim(bottom=0);axes[axis,col].set_xlabel('Executed environment step; >=5 tasks')
   axes[0,col].set_title(suite);axes[0,col].set_ylabel('Position vector bias / scatter (mm)');axes[1,col].set_ylabel('World SO(3) log bias / scatter (deg)')
  axes[0,0].legend(fontsize=7);fig.suptitle(title+' | solid: norm(mean signed error); dashed: RMS scatter\nDescriptive paired rollouts; feedback/state divergence included.');save(fig,'02-'+tag+'-bias-scatter')
  fig,axes=plt.subplots(2,4,figsize=(17,8),constrained_layout=True)
  for col,suite in enumerate(suites):
   for c in group:
    rows=rows_for(suite,c)
    for axis,kind in enumerate(['first_translation_by_position','first_rotation_by_position']):
     grid=np.full((20,8),np.nan)
     for i,row in enumerate(rows):grid[i,:len(row[kind])]=row[kind]
     med=np.nanmedian(grid,0);lo,hi=np.nanpercentile(grid,[25,75],axis=0);axes[axis,col].plot(range(8),med,color=colors[c],marker='.',label=labels[c]);axes[axis,col].fill_between(range(8),lo,hi,color=colors[c],alpha=.1)
   axes[0,col].set_title(suite)
   for axis in range(2):axes[axis,col].set_ylim(bottom=0);axes[axis,col].grid(alpha=.15);axes[axis,col].set_xlabel('Executed position in first8-action chunk')
   axes[0,col].set_ylabel('Translation command-channel RMSE');axes[1,col].set_ylabel('Rotation command-channel RMSE')
  axes[0,0].legend(fontsize=7);fig.suptitle(title+' | exactly same first observation, original command units\nMedian + trajectory IQR. These command errors are not millimetre / SO(3) pose errors.');save(fig,'03-'+tag+'-same-input-chunk')
 keys=[(s,t,j) for s in suites for t in range(10) for j in [5,15]];lookup={(x['suite'],x['task_id'],x['reset_index'],x['condition']):x for x in metrics};fig,axes=plt.subplots(1,3,figsize=(17,20),constrained_layout=True)
 for ax,field,title in zip(axes[:2],['translation_p95_mm','rotation_p95_deg'],['log10(1+ position P95 mm)','log10(1+ SO(3) P95 deg)']):
  grid=np.array([[lookup[(*key,c)][field] for c in conditions] for key in keys]);im=ax.imshow(np.log10(1+grid),aspect='auto',interpolation='nearest',cmap='viridis');ax.set_title(title);ax.set_xticks(range(6));ax.set_xticklabels([labels[c] for c in conditions],rotation=55,ha='right',fontsize=8);ax.set_yticks(range(80));ax.set_yticklabels([f"{s.replace('libero_','')}/{t}/i{j}" for s,t,j in keys],fontsize=6);fig.colorbar(im,ax=ax,shrink=.35)
 success=np.array([[lookup[(*key,conditions[0])]['reference_success']]+[lookup[(*key,c)]['success'] for c in conditions] for key in keys],dtype=int);axes[2].imshow(success,aspect='auto',interpolation='nearest',vmin=0,vmax=1,cmap='RdYlGn');axes[2].set_xticks(range(7));axes[2].set_xticklabels(['BF16']+[labels[c] for c in conditions],rotation=55,ha='right',fontsize=8);axes[2].set_yticks(range(80));axes[2].set_yticklabels([f"{s.replace('libero_','')}/{t}/i{j}" for s,t,j in keys],fontsize=6);axes[2].set_title('Observed success: red0 /green1')
 for ax in axes:
  for n in [19.5,39.5,59.5]:ax.axhline(n,color='white',lw=1)
 fig.suptitle('All80 matched cases |40 development tasks, two initial states | no cross-model inference');save(fig,'04-case-error-success-map')
 fig,axes=plt.subplots(2,4,figsize=(17,8),constrained_layout=True)
 for col,suite in enumerate(suites):
  for ci,c in enumerate(conditions):
   rows=rows_for(suite,c);jitter=np.linspace(-.17,.17,len(rows))
   for axis,field in enumerate(['translation_windows','rotation_windows']):
    for i,row in enumerate(rows):
     observed=bool(row[field]);y=row[field][0]['confirmed_index'] if observed else row['common_steps']-1
     axes[axis,col].scatter(ci+jitter[i],y,color=colors[c] if observed else 'none',edgecolor=colors[c],marker='o' if observed else 'v',s=14,alpha=.7)
  for axis in range(2):axes[axis,col].set_xticks(range(6));axes[axis,col].set_xticklabels([labels[c] for c in conditions],rotation=55,ha='right',fontsize=7);axes[axis,col].set_ylabel('Confirmation step / last common step');axes[axis,col].grid(alpha=.15)
  axes[0,col].set_title(suite)
 fig.suptitle('Frozen5mm /2deg exceedance lasting5 steps | circle: first confirmation, hollow triangle: no event in prefix\nNot a geometric change-point; early success termination is informative censoring.');save(fig,'05-persistent-window-confirmation')
 fig=plt.figure(figsize=(18,10),constrained_layout=True)
 for row,reset in enumerate([5,15]):
  for col,suite in enumerate(suites):
   ax=fig.add_subplot(2,4,row*4+col+1,projection='3d');n=min(len(traces[suite,0,reset,c]['translation_mm']) for c in conditions);ref=baseline_positions[suite,0,reset][:n];cloud=[ref]
   ax.plot(*ref.T,color='#999999',ls='--',lw=2.5,label='BF16')
   for c in conditions:
    xyz=ref+traces[suite,0,reset,c]['translation_vector_mm'][:n];cloud.append(xyz);ax.plot(*xyz.T,color=colors[c],lw=1.2,label=labels[c]);ax.scatter(*xyz[-1],color=colors[c],s=8)
   pts=np.concatenate(cloud);center=(pts.min(0)+pts.max(0))/2;half=max(float(np.ptp(pts,axis=0).max()/2),1.)*1.05
   ax.set_xlim(center[0]-half,center[0]+half);ax.set_ylim(center[1]-half,center[1]+half);ax.set_zlim(center[2]-half,center[2]+half);ax.set_box_aspect([1,1,1]);ax.set_title(f'{suite} /task0 /initial{reset}\n{n} shared steps',fontsize=10);ax.set_xlabel('Base X (mm)');ax.set_ylabel('Base Y (mm)');ax.set_zlabel('Base Z (mm)')
   if row==0 and col==0:ax.legend(fontsize=6)
 fig.suptitle('Prespecified task0, both initial states | equal XYZ scale within each panel\nAll7 paths share the same observed control-step prefix; endpoint markers are not contact labels.');save(fig,'06-task0-two-initials-3d-paths')
 (out/'risk-set-coverage.json').write_text(json.dumps(dict(display_policy='At least5 distinct tasks; repeated initial states are not independent tasks; no tail padding or time warp',coverage=coverage),indent=2)+'\n')
 html='<meta charset="utf-8"><title>OFT七条件轨迹漂移</title><h1>OpenVLA-OFT七条件、80病例配对</h1><p>BF16与六个量化配置；四suite、十任务、初态5/15。不是五初态或三模型完成。曲线含闭环反馈，窗口不是几何拐点或已验证补偿触发点；没有训练/OPD干预。<a href="summary.json">统计与逐对指标</a></p>'
 for name in files:html+=f'<h2>{name}</h2><a href="{name}.pdf">PDF</a><img style="width:100%" src="{name}.png">'
 (out/'index.html').write_text(html,encoding='utf-8')
