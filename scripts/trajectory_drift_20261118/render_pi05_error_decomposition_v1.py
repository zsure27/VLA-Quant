"""Aggregate world-frame error center/spread; individual trajectories stay private."""
import argparse,json,os,sys
from pathlib import Path
repo=Path(__file__).resolve().parents[2];sys.path.insert(0,str(repo/'.local/report-libs'));os.environ.setdefault('MPLCONFIGDIR',str(repo/'.local/matplotlib-cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
data=json.loads(a.source.read_text());assert data['gate']=='PASS_PAIRED_VECTOR_CENTER_SPREAD_RMS_IDENTITY'
colors={'smoothquant_w4a8':'#6556ae','gptq_w4a16':'#238a83','awq_w4a16':'#df8c26','omniquant_w4a16':'#3b7fb5'}
names={'smoothquant_w4a8':'SmoothQuant W4A8','gptq_w4a16':'GPTQ W4A16','awq_w4a16':'AWQ W4A16','omniquant_w4a16':'OmniQuant LWC W4A16'}
suites=['libero_spatial','libero_object','libero_goal','libero_10']
fields=[('position_world_mm','center_magnitude','Position center (mm)'),('position_world_mm','rms_dispersion','Position RMS spread (mm)'),('rotation_world_log_deg','center_magnitude','Rotation center (deg)'),('rotation_world_log_deg','rms_dispersion','Rotation RMS spread (deg)')]
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(5,4,figsize=(16,12),sharex='col',constrained_layout=True,gridspec_kw={'height_ratios':[3,3,3,3,1]})
for column,suite in enumerate(suites):
    for row in data['curves']:
        c=row['condition']
        if row['suite']!=suite or c not in colors:continue
        for i,(field,key,_) in enumerate(fields):
            values=row[field];y=np.asarray([np.nan if v is None else v for v in values[key]])
            assert all(v is None or values['n'][j]>=5 for j,v in enumerate(values[key]))
            axes[i,column].plot(row['step'],y,color=colors[c],lw=1.5,label=names[c])
        axes[4,column].step(row['step'],row['position_world_mm']['n'],where='post',color=colors[c])
    axes[0,column].set_title(suite)
    for i in range(5):axes[i,column].grid(alpha=.17);axes[i,column].set_ylim(bottom=0)
    axes[4,column].set_xlabel('Executed control step')
for i,(_,_,label) in enumerate(fields):axes[i,0].set_ylabel(label)
axes[4,0].set_ylabel('Observed pairs N');axes[0,0].legend(fontsize=8)
fig.suptitle('Independent PyTorch pi0.5 | initial 5 | cross-case error-vector center and spread\nWorld-frame SO(3) log, N>=5; closed-loop descriptive separation, not same-state noise or causal attribution',fontsize=12)
a.output.parent.mkdir(parents=True,exist_ok=True);assert not a.output.exists();fig.savefig(a.output,dpi=160);plt.close(fig)
print('CENTER_SPREAD_AGGREGATE_FIGURE_SAVED',a.output)
