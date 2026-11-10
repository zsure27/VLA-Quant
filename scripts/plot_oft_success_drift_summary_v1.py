import pathlib,json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=pathlib.Path('analysis/20261110-oft-five-methods-reset5-v1');d=json.load(open(r/'summary.json'));keys=['awq_w4a16','gptq_w4a16','omniquant_w4a16','smoothquant_w4a8','smoothquant_w4a4'];labels=['AWQ W4A16','GPTQ W4A16','Omni LWC W4A16','SQ W4A8','SQ W4A4'];colors=['#2878b5','#268c70','#7956a3','#e07b39','#b44770'];rows=[]
for k in keys:
 a=[x for x in d['metrics'] if x['condition']==k];rows.append({'method':k,'pairs':len(a),'success':sum(x['success'] for x in a),'rescue':sum(x['success'] and not x['reference_success'] for x in a),'break':sum(x['reference_success'] and not x['success'] for x in a),'median_task_position_p95_mm':float(np.median([x['translation_p95_mm'] for x in a])),'median_task_rotation_p95_deg':float(np.median([x['rotation_p95_deg'] for x in a]))})
fig,axes=plt.subplots(1,3,figsize=(16,5));sr=[100*x['success']/40 for x in rows];axes[0].bar(range(5),sr,color=colors);axes[0].axhline(92.5,color='#444444',ls='--',label='BF16 37/40');axes[0].set_ylim(0,110);axes[0].set_title('Task success (%)');axes[0].legend(fontsize=8)
for i,x in enumerate(rows):axes[0].text(i,sr[i]+2,str(x['success'])+'/40',ha='center',fontsize=9)
for ax,field,title in zip(axes[1:],['median_task_position_p95_mm','median_task_rotation_p95_deg'],['Position deviation (mm)','Rotation deviation (degrees)']):
 values=[x[field] for x in rows];ax.bar(range(5),values,color=colors);ax.set_title(title+'\nMedian of 40 per-task P95 errors');ax.set_ylim(0,max(values)*1.18)
 for i,v in enumerate(values):ax.text(i,v+max(values)*.025,f'{v:.1f}',ha='center',fontsize=9)
for ax in axes:ax.set_xticks(range(5));ax.set_xticklabels(labels,rotation=25,ha='right',fontsize=9);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('MEASURED OpenVLA-OFT | 4 suites x 10 tasks | initial5 only | no causal method ranking');fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(r/('08-success-and-drift-summary.'+ext),dpi=160,bbox_inches='tight')
(r/'aggregate-small.json').write_text(json.dumps(rows,indent=2));page=r/'index.html';page.write_text(page.read_text()+'<h2>Success and drift summary</h2><img style="width:100%" src="08-success-and-drift-summary.png">');print('SUMMARY_FIGURE_SAVED')
