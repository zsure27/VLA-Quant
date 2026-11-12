"""Plot only audited, completed pilot pairs; no padding or time warping."""
import argparse,json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--curves',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
rows=json.loads(a.curves.read_text());assert len(rows)==4
fig,axes=plt.subplots(2,4,figsize=(15,6.3),squeeze=False)
for col,row in enumerate(rows):
 for line,(key,label,color) in enumerate([('translation_mm','Position separation (mm)','#0072B2'),('rotation_deg','Orientation separation (deg)','#D55E00')]):
  v=np.asarray(row[key]);assert len(v)>0 and np.isfinite(v).all()
  ax=axes[line,col];ax.plot(np.arange(len(v)),v,color=color,lw=1.5)
  ax.set_title(f"{row['suite']} | task {row['task_id']}, reset {row['reset_index']}\nn=1 pair; {len(v)} common steps",fontsize=10)
  ax.set_xlabel('Executed environment step');ax.set_ylabel(label);ax.grid(alpha=.2);ax.set_xlim(0,max(len(v)-1,1));ax.set_ylim(bottom=0)
fig.suptitle('OpenVLA-OFT: audited QVLA mixed-channel W~4 / BF16-activation pilot vs BF16',fontsize=14,y=.99)
fig.text(.5,.025,'Free rollouts at matched environment steps; includes diverging observations. Different y scales.\n4 successful pairs only; no shared change-point or causal mechanism established.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.09,1,.95));a.output.mkdir(parents=True,exist_ok=True)
for suffix in ['png','pdf']:fig.savefig(a.output/('qvla-pilot-drift.'+suffix),dpi=180)
plt.close(fig)
