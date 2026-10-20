"""Visualize real frozen-profile shapes, separately from rollout performance."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
profiles=json.loads((HERE/"awq-profile-metadata.json").read_text(encoding="utf-8"))
assert len(profiles)==3 and all(p["entry_count"]==422 for p in profiles)
assert all(p["scope_summary"]==profiles[0]["scope_summary"] for p in profiles)
counts=profiles[0]["scope_summary"];total=sum(v["weight_parameters"] for v in counts.values())
rows=[dict(branch=k,targets=v["targets"],weight_parameters=v["weight_parameters"],weight_fraction=v["weight_parameters"]/total,clip_targets=v["clip_targets"]) for k,v in counts.items()]
(HERE/"data").mkdir(exist_ok=True);(HERE/"figures").mkdir(exist_ok=True)
with (HERE/"data/awq_scope.csv").open("w",newline="",encoding="utf-8") as stream:
 writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
fig,axes=plt.subplots(1,2,figsize=(9,3.5));labels=[r["branch"] for r in rows];colors=["#5277A6","#A7725B","#6E957A"]
for ax,values,title in [(axes[0],[r["targets"] for r in rows],"Target module count"),(axes[1],[100*r["weight_fraction"] for r in rows],"Share of target weight parameters (%)")]:
 bars=ax.barh(labels,values,color=colors);ax.invert_yaxis();ax.set_title(title);ax.set_xlim(0,max(values)*1.22)
 for bar,value in zip(bars,values):ax.text(value+max(values)*.02,bar.get_y()+bar.get_height()/2,"{:.1f}".format(value),va="center")
fig.suptitle("Frozen AWQ profile: 422 targets; 32 calibration samples",fontsize=11);fig.tight_layout(rect=[0,0,1,.92])
for ext in ("png","svg"):fig.savefig(HERE/"figures"/("00_awq_scope."+ext),dpi=160)
plt.close(fig)
result=dict(connected_target_weight_parameters=total,language_fraction=counts["language"]["weight_parameters"]/total,vision_fraction=(counts["DINO"]["weight_parameters"]+counts["SigLIP"]["weight_parameters"])/total,note="Only quantization-target weights. Excludes protected modules and metadata; fake quant is still BF16 storage. Not a realized compression ratio.")
(HERE/"data/awq_scope_summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result))
