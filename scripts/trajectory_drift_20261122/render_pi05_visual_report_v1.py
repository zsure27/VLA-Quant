# -*- coding: utf-8 -*-
"""Render audited Pi0.5 geometry into figures and a self-contained Chinese HTML report."""
import argparse,base64,json,html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # Register projection on older plotting runtimes.

def main():
 p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--dispositions',type=Path);p.add_argument('--sq4-exclusions',type=Path);a=p.parse_args()
 def load(n):return json.loads((a.analysis/n).read_text(encoding='utf-8'))
 summary=load('summary-public.json');curves=load('curves-public.json');examples=load('examples-private.json');decomp=load('error-decomposition-public.json')
 assert summary['gate']=='PASS_COMPLETE_200_PI05_SAME_BACKEND_RAW_GEOMETRY_AUDIT' and not summary['missing']
 assert decomp['gate']=='PASS_PAIRED_VECTOR_CENTER_SPREAD_RMS_IDENTITY'
 a.output.mkdir(parents=True,exist_ok=False)
 conditions=list(summary['coverage']);suites=summary['suites']
 labels={'bf16':'BF16','gptq_w4a16':'GPTQ W4A16','awq_w4a16':'AWQ W4A16','omniquant_w4a16':'OmniQuant LWC W4A16','smoothquant_w4a8':'SmoothQuant W4A8','duquant_w4a8':'DuQuant W4A8','duquant_w4a4':'DuQuant W4A4'}
 palette=['#202938','#2563eb','#db6b19','#8b5cf6','#0891b2','#c73564','#469228'];colors=dict(zip(conditions,palette))
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
 figures=[]
 def save(fig,name,caption):
  fig.savefig(a.output/name,dpi=160,bbox_inches='tight');plt.close(fig);figures.append((name,caption))
 fig,axs=plt.subplots(1,2,figsize=(14,5))
 counts=[summary['coverage'][c]['success'] for c in conditions];names=[labels[c] for c in conditions]
 axs[0].barh(names,np.array(counts)/2,color=[colors[c] for c in conditions]);axs[0].invert_yaxis();axs[0].set(xlim=(0,115),xlabel='Success rate (%)',title='All 200 episodes per configuration')
 for i,v in enumerate(counts):axs[0].text(v/2+1,i,f'{v}/200',va='center',fontsize=8)
 table=np.array([[summary['coverage'][c]['suites'][s]['success']/50*100 for s in suites] for c in conditions])
 im=axs[1].imshow(table,vmin=0,vmax=100,cmap='Blues',aspect='auto');axs[1].set_xticks(range(4));axs[1].set_xticklabels(['Spatial','Object','Goal','Long']);axs[1].set_yticks(range(len(conditions)));axs[1].set_yticklabels(['']*len(names));axs[1].set_title('Per-suite success rate (same row order, 50 each)')
 for i in range(len(conditions)):
  for j in range(4):axs[1].text(j,i,f'{table[i,j]:.0f}%',ha='center',va='center',color='white' if table[i,j]>60 else 'black',fontsize=8)
 fig.colorbar(im,ax=axs[1],label='%');fig.tight_layout();save(fig,'01_success.png','成功率只反映任务终态，不代表轨迹与BF16一致。每配置200条，每套件50条。')
 fig=plt.figure(figsize=(18,14))
 for row,(view,axes) in enumerate([('3D',None),('XY',(0,1)),('XZ',(0,2)),('YZ',(1,2))]):
  for col,suite in enumerate(suites):
   ax=fig.add_subplot(4,4,row*4+col+1,projection='3d' if axes is None else None)
   for c in conditions:
    item=[x for x in examples if x['condition']==c and x['suite']==suite];assert len(item)==1
    xyz=np.asarray(item[0]['position_m']);kw=dict(color=colors[c],label=labels[c],linewidth=1.1,alpha=.9)
    if axes is None:
     ax.plot(*xyz.T,**kw);ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)')
     for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.set_major_locator(MaxNLocator(4))
     ax.tick_params(labelsize=8)
    else:
     i,j=axes;ax.plot(xyz[:,i],xyz[:,j],**kw);ax.scatter(xyz[-1,i],xyz[-1,j],s=13,color=colors[c],marker='x');ax.set(xlabel='xyz'[i]+' (m)',ylabel='xyz'[j]+' (m)');ax.set_aspect('equal',adjustable='datalim')
    ax.grid(alpha=.2)
   ax.set_title(suite.replace('libero_','').replace('10','Long')+' / '+view)
 handles,ls=ax.get_legend_handles_labels();fig.legend(handles,ls,loc='upper center',ncol=4,bbox_to_anchor=(.5,.977));fig.suptitle('Fixed examples: task 0 / initial 5 in each suite; full paths to own termination',y=.998);fig.tight_layout(rect=(0,0,1,.95));save(fig,'02_geometry_3d_2d.png','固定展示各suite的task0、初态5，未按漂移大小或成败挑选。显示各自完整轨迹；二维叉号为终点，路径长度不同不补齐。示例不能单独证明共性。')
 fig,axs=plt.subplots(3,4,figsize=(18,11),squeeze=False)
 for col,suite in enumerate(suites):
  for c in conditions[1:]:
   row=next(x for x in curves if x['condition']==c and x['suite']==suite);x=row['step']
   for ri,key in enumerate(['position_mm','rotation_deg']):
    val=row[key];ax=axs[ri,col];ax.plot(x,val['median'],color=colors[c],label=labels[c],lw=1.2);ax.fill_between(x,np.asarray(val['p25'],dtype=float),np.asarray(val['p75'],dtype=float),color=colors[c],alpha=.07)
   axs[2,col].plot(x,row['position_mm']['n'],color=colors[c],label=labels[c])
  axs[0,col].set_title(suite.replace('libero_','').replace('10','Long'));axs[2,col].set_xlabel('Executed control step (paired common prefix)')
  for ri in range(3):axs[ri,col].grid(alpha=.2)
 axs[0,0].set_ylabel('Position deviation (mm)');axs[1,0].set_ylabel('SO(3) deviation (deg)');axs[2,0].set_ylabel('Paired episodes contributing (N)');axs[0,0].legend(fontsize=7)
 fig.tight_layout();save(fig,'03_error_curves.png','按同一控制步对齐：实线为各病例误差中位数，阴影为25%–75%分位区间，并显示实际有效配对数N。任一侧结束即停止配对，N<5隐藏。后段样本组成变化，不能直接将曲线上扬解释为量化误差累积。')
 fig,axs=plt.subplots(4,4,figsize=(18,13),squeeze=False)
 for col,suite in enumerate(suites):
  for c in conditions[1:]:
   row=next(x for x in decomp['curves'] if x['condition']==c and x['suite']==suite)
   for ri,(field,metric) in enumerate([('position_world_mm','center_magnitude'),('position_world_mm','rms_dispersion'),('rotation_world_log_deg','center_magnitude'),('rotation_world_log_deg','rms_dispersion')]):axs[ri,col].plot(row['step'],row[field][metric],color=colors[c],label=labels[c])
  axs[0,col].set_title(suite.replace('libero_','').replace('10','Long'))
  for ri in range(4):axs[ri,col].grid(alpha=.2);axs[ri,col].set_xlabel('Control step')
 for ax,label in zip(axs[:,0],['Position center (mm)','Position spread RMS (mm)','Rotation center (deg)','Rotation spread RMS (deg)']):ax.set_ylabel(label)
 axs[0,0].legend(fontsize=7);fig.tight_layout();save(fig,'04_center_spread.png','将配对误差向量分为均值中心幅度和离散RMS，并核验总RMS平方＝中心幅度平方＋离散RMS平方。旋转采用世界坐标下SO(3)主值对数。这里只描述跨病例闭环差异，不能等同于同状态量化噪声的均值和方差。')
 fig,axs=plt.subplots(2,4,figsize=(18,9),squeeze=False)
 for ri,metric in enumerate(['position_p95_mm','rotation_p95_deg']):
  data=[]
  for suite in suites:
   grid=np.array([[np.median([x[metric] for x in summary['pairs'] if x['condition']==c and x['suite']==suite and x['task_id']==t]) for t in range(10)] for c in conditions[1:]])
   assert np.isfinite(grid).all();data.append(grid)
  vmax=max(float(x.max()) for x in data)
  for col,(suite,grid) in enumerate(zip(suites,data)):
   im=axs[ri,col].imshow(grid,vmin=0,vmax=vmax,cmap='magma',aspect='auto');axs[ri,col].set(xticks=range(10),xlabel='Task ID',title=suite.replace('libero_','').replace('10','Long'));axs[ri,col].set_yticks(range(len(conditions)-1));axs[ri,col].set_yticklabels([labels[c] for c in conditions[1:]] if col==0 else ['']*(len(conditions)-1));fig.colorbar(im,ax=axs[ri,col],label='mm' if ri==0 else 'deg')
 fig.suptitle('Per task: median of five episode-wise p95 deviations');fig.tight_layout();save(fig,'05_task_heatmaps.png','每个格子先计算单轨迹共同前缀的95%分位误差，再对同任务5个初态取中位数。同一指标四套件使用相同色标；用于检查跨任务一致性，不将单条峰值当共同拐点。')
 # Fixed early-step comparison; full N=50 at both points prevents censoring-composition changes.
 early=[]
 for c in conditions[1:]:
  for suite in suites:
   v=next(x for x in curves if x['condition']==c and x['suite']==suite)
   for field in ['position_mm','rotation_deg']:
    z=v[field];eligible=len(v['step'])>=50 and z['n'][9]==50 and z['n'][49]==50 and z['median'][9] is not None and z['median'][49] is not None
    early.append(dict(condition=c,suite=suite,metric=field,eligible_full_50=eligible,n10=z['n'][9] if len(v['step'])>=10 else None,n50=z['n'][49] if len(v['step'])>=50 else None,median10=z['median'][9] if eligible else None,median50=z['median'][49] if eligible else None,delta=z['median'][49]-z['median'][9] if eligible else None))
 (a.output/'early-step-comparison.json').write_text(json.dumps(early,allow_nan=False,indent=2))
 fig,axs=plt.subplots(2,4,figsize=(18,8),squeeze=False)
 for ri,field in enumerate(['position_mm','rotation_deg']):
  vals=[x['delta'] for x in early if x['metric']==field and x['eligible_full_50']];lo=min([0]+vals);hi=max([1]+vals)
  for col,suite in enumerate(suites):
   values=[next(x for x in early if x['condition']==c and x['suite']==suite and x['metric']==field) for c in conditions[1:]]
   ax=axs[ri,col];numbers=[x['delta'] if x['eligible_full_50'] else np.nan for x in values];ax.barh(range(len(numbers)),numbers,color=[colors[c] for c in conditions[1:]])
   ax.set_yticks(range(len(numbers)));ax.set_yticklabels([labels[c] for c in conditions[1:]] if col==0 else ['']*len(numbers));ax.set_xlim(lo-(hi-lo)*.1,hi+(hi-lo)*.2);ax.axvline(0,color='#666666',lw=.7);ax.grid(axis='x',alpha=.2);ax.set_xlabel('Median error at step 50 minus step 10 ('+('mm' if ri==0 else 'deg')+')')
   if ri==0:ax.set_title(suite.replace('libero_','').replace('10','Long'))
   for j,x in enumerate(values):
    if x['eligible_full_50']:ax.text(x['delta'],j,' %.2f'%x['delta'],va='center',fontsize=8)
 fig.suptitle('Fixed early-step comparison | same 50 episodes at both steps | descriptive only');fig.tight_layout(rect=(0,0,1,.96));save(fig,'06_early_growth.png','固定比较第10与第50个控制步的误差中位数差值，仅在两个步点都保留全部50例时展示，避免因提前结束造成样本组成变化。每格并非独立实验，正值不等于证明了突变点或纯量化误差累积。')
 rows=[]
 for c in conditions:
  cov=summary['coverage'][c];paired=[x for x in summary['pairs'] if x['condition']==c];rescue=sum(x['success'] and not x['reference_success'] for x in paired);brk=sum(not x['success'] and x['reference_success'] for x in paired)
  pos=np.median([x['position_p95_mm'] for x in paired]) if paired else 0;rot=np.median([x['rotation_p95_deg'] for x in paired]) if paired else 0
  rows.append('<tr>'+''.join(f'<td>{v}</td>' for v in [html.escape(labels[c]),f"{cov['success']}/200",f"{cov['success']/2:.1f}%",*[f"{cov['suites'][s]['success']}/50" for s in suites],f'{pos:.2f}',f'{rot:.2f}',str(rescue) if paired else '—',str(brk) if paired else '—'])+'</tr>')
 failures=[{k:x[k] for k in ['condition','suite','task_id','reset_index']} for x in summary['records'] if not x['success']]
 (a.output/'failures.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2))
 if a.sq4_exclusions:
  excluded=json.loads(a.sq4_exclusions.read_text(encoding='utf-8'));assert excluded['complete']==80 and excluded['successes']==0 and len(excluded['rows'])==80
  (a.output/'excluded-sq4-failures.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2),encoding='utf-8')
 complete_final='duquant_w4a4' in conditions
 title='π0.5量化轨迹漂移可视化实验报告' if complete_final else 'π0.5量化轨迹漂移阶段报告（DuQuant W4A4待补齐）'
 sections=[]
 for name,caption in figures:
  enc=base64.b64encode((a.output/name).read_bytes()).decode();sections.append(f'<figure><img src="data:image/png;base64,{enc}"><figcaption>{html.escape(caption)}</figcaption></figure>')
 doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'+title+'</title><style>body{font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif;max-width:1500px;margin:30px auto;padding:0 24px;color:#17283b}h1,h2{line-height:1.4}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border:1px solid #ccd6df;text-align:center}th{background:#eaf0f6}img{width:100%;height:auto}figure{margin:28px 0}figcaption{color:#526275;font-size:14px}.note{background:#edf4fa;padding:15px;border-left:4px solid #286da7}code{font-size:13px}</style><h1>'+title+'</h1>'
 doc+='<p class="note">独立PyTorch π0.5队列：4个LIBERO suite × 每suite 10任务 × 每任务5初态＝每配置200条轨迹。模型seed7、环境seed0、相同初始观测与查询噪声；先生成10步动作块，执行前5步。原生JAX与未通过的JAX→PyTorch桥接不并入本表。W4/A4等标签指实际profile所覆盖模块的目标精度；未量化模块保留原后端精度。</p>'
 doc+='<h2>结果与简要分析</h2><p>本实验观察量化前后闭环轨迹的偏移，不进行量化方法改进或OPD训练。成功率、轨迹位置偏差、旋转偏差分别报告：任务成功并不意味着路径没有偏移。是否存在可补偿的共性区间，应同时查看四套件的误差曲线、有效样本数和逐任务热力图。</p>'
 posstats=[];rotstats=[];long_highest=0
 for c in conditions[1:]:
  paired=[x for x in summary['pairs'] if x['condition']==c]
  posstats.append(float(np.median([x['position_p95_mm'] for x in paired])));rotstats.append(float(np.median([x['rotation_p95_deg'] for x in paired])))
  suitepos={suite:float(np.median([x['position_p95_mm'] for x in paired if x['suite']==suite])) for suite in suites}
  long_highest+=max(suitepos,key=suitepos.get)=='libero_10'
 doc+=f'<p><b>当前可确认的现象：</b>完整量化配置的成功率为{min(summary["coverage"][c]["success"]/2 for c in conditions[1:]):.1f}%–{max(summary["coverage"][c]["success"]/2 for c in conditions[1:]):.1f}%，但其单轨迹p95位置偏差的跨病例中位数仍为{min(posstats):.1f}–{max(posstats):.1f}毫米，旋转偏差为{min(rotstats):.1f}–{max(rotstats):.1f}度。因此只看任务成功率，会漏掉不同量化配置的控制路径差异。</p>'
 doc+=f'<p><b>跨套件的共性候选：</b>{long_highest}/{len(conditions)-1}个量化配置在Long套件上取得四套件中最高的典型位置偏差。它说明长任务队列值得重点观察，但不同套件任务、执行长度及后期有效样本不同，不能单独据此断言是误差随时间累积造成。本轮尚未证明所有方法、所有任务共享同一个时间拐点。</p>'
 for field,label in [('position_mm','平移'),('rotation_deg','旋转')]:
  valid=[x for x in early if x['metric']==field and x['eligible_full_50']];positive=sum(x['delta']>0 for x in valid)
  if valid:doc+=f'<p><b>{label}偏移的描述性共性：</b>在第10和第50步均保留全部50例的{len(valid)}个“配置×套件”组合中，{positive}个组合的第50步误差中位数更大。这是固定早期区间的闭环分离现象，不代表所有单轨迹单调增长，也不是共同拐点的显著性检验。</p>'
 doc+='<table><thead><tr>'+''.join('<th>'+x+'</th>' for x in ['配置','成功/总数','成功率','Spatial','Object','Goal','Long','轨迹p95位置偏差中位数(mm)','轨迹p95旋转偏差中位数(°)','相对BF16救回','相对BF16新增失败'])+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
 doc+='<p>表中误差以同任务、同初态BF16为参照，仅计算两侧共同执行前缀；先取每条轨迹的p95，再跨200条取中位数。救回和新增失败均为真实配对计数。</p><h2>综合可视化</h2>'+''.join(sections)
 doc+='<h2>SmoothQuant W4A4：单独说明</h2><p>历史已测试80条均未成功（0/80），用户已取消剩余部分。该配置与本表200条完整队列覆盖不同，因此从综合轨迹图中剔除，不能以0/80与其他200条直接作同覆盖排名。其失败现象提示该实现与标定配方在4bit激活下存在严重退化；仅凭失败率还不能判定是激活离群值、层敏感度或后续控制反馈中的哪一项造成。</p>'
 if a.dispositions:
  dispositions=json.loads(a.dispositions.read_text(encoding='utf-8'))
  blocked=[x for x in dispositions['rows'] if x.get('full_method_admitted') is False]
  assert len(blocked)==5
  explanations={'qvla_w4a16':'完整动作敏感度、通道分配、量化注入与重载及pilot尚未接入；动作图前置门禁通过不等于完整方法完成。','actquant_w4a16':'原生C++/GGUF同检查点输入与动作桥接、HSIC/Fisher分配未完成，当前完整流程的资源预算未满足。','quantvla_w4a8':'已有原生GR00T ATM/OHB源码，但π0.5 flow后端适配、数值门禁和pilot未完成。','quantvla_w4a4':'除同样的π0.5 flow适配缺口外，还需要独立A4数值门禁。','holoq_w4a4':'候选源码与HoloQ名称、原配置来源尚未完成一手来源确认，未用其他量化器改名代替。'}
  doc+='<h2>尚未完成的配置</h2><p>以下5项仍有适配、来源或资源阻断，不记为已完成，也不等同于已经证明其架构不适用。综合曲线只覆盖本轮真实跑通的配置；原计划未被改写成“全部12项量化均完成”。</p><table><tr><th>配置</th><th>当前缺口</th></tr>'
  for row in blocked:doc+='<tr><td>'+html.escape(row['condition'])+'</td><td>'+explanations[row['condition']]+'</td></tr>'
  doc+='</table>'
 doc+='<h2>结论的边界与后续可用性</h2><ul><li>本队列采用fake quant，用于测量控制偏移；不能声称实现真实INT4内核加速或压缩后部署显存。</li><li>OmniQuant仅接入LWC，未实现LET；其结论只适用于该变体。</li><li>闭环后续观测会分叉，误差同时含量化动作差异和环境反馈。按相同控制步对齐还会混合空间路径差异与执行快慢差异；本轮没有做动作阶段或动态时间规整对齐，不能把全部偏差当作动作头的同状态预测误差。</li><li>不将阈值越界、轨迹曲线的视觉转折或夹爪命令等同于统计变点、真实接触事件或已经证明的共同拐点。</li><li>OPD补偿效果尚未测试；本报告只能提供后续同状态教师对照和补偿区间验证的候选证据。优先验证跨套件、跨方法重复出现的分离区间，再判断补偿是否有效。</li></ul><p>主表配置的全部失败病例见同目录failures.json；被取消的SQ W4A4历史失败另列excluded-sq4-failures.json，不混入主表分母。来源审计与数值汇总随报告保留。本文不冒称尚未适配的方法已完成，也不混合其他模型的分母。</p></html>'
 (a.output/'π0.5量化轨迹漂移实验报告.html').write_text(doc,encoding='utf-8')
 for name in ['summary-public.json','curves-public.json','error-decomposition-public.json']:(a.output/name).write_bytes((a.analysis/name).read_bytes())
 print(json.dumps(dict(report=str(a.output/'π0.5量化轨迹漂移实验报告.html'),figures=len(figures),configurations=len(conditions),full_final=complete_final),ensure_ascii=False))

if __name__=='__main__':main()
