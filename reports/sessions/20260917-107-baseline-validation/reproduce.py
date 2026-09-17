import json,csv,math,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2];RAW=REPO/"results/107-baseline-validation-20260917"
files=sorted(RAW.glob("awq-baseline-shard-*/paired-results.json"))
if not files:raise SystemExit("No completed paired shards; no fabricated figure")
pairs=[];contracts=set()
for p in files:
 d=json.loads(p.read_text(encoding="utf-8"));pairs.extend(d["paired_episodes"])
 contracts.add((p.parent/"CONTRACT_SHA256SUMS.txt").read_text(encoding="utf-8"))
assert len(contracts)==1,"source/profile hashes differ"
keys=[(r["task_id"],r["init_state_index"]) for r in pairs];assert len(keys)==len(set(keys))
pairs.sort(key=lambda r:(r["task_id"],r["init_state_index"]))
(HERE/"data").mkdir(exist_ok=True);(HERE/"figures").mkdir(exist_ok=True)
summary=[]
for case in ("bf16","w4"):
 summary.append(dict(case=case,episodes=len(pairs),successes=sum(r[case+"_success"] for r in pairs),success_rate=sum(r[case+"_success"] for r in pairs)/len(pairs),episode_errors=sum(len(r[case+"_episode_errors"]) for r in pairs)))
rows=[]
for task in range(10):
 rs=[r for r in pairs if r["task_id"]==task]
 rows.append(dict(task_id=task,episodes=len(rs),bf16_successes=sum(r["bf16_success"] for r in rs),w4_successes=sum(r["w4_success"] for r in rs)))
for name,values in [("paired_episodes",pairs),("summary",summary),("per_task",rows)]:
 with (HERE/"data"/(name+".csv")).open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
only_b=sum(r["bf16_success"] and not r["w4_success"] for r in pairs);only_w=sum(r["w4_success"] and not r["bf16_success"] for r in pairs);n=only_b+only_w
def comb(n,k):return math.factorial(n)//(math.factorial(k)*math.factorial(n-k))
result=dict(variants=summary,covered_initial_indices=sorted(set(r["init_state_index"] for r in pairs)),paired_episodes=len(pairs),bf16_only=only_b,w4_only=only_w,both_success=sum(r["bf16_success"] and r["w4_success"] for r in pairs),both_failure=sum(not r["bf16_success"] and not r["w4_success"] for r in pairs),difference_percentage_points=100*(only_w-only_b)/len(pairs),mcnemar_exact_p=min(1.,2*sum(comb(n,k) for k in range(min(only_b,only_w)+1))/2**n) if n else 1.,note="Fixed tasks, paired official init indices; p value not equivalence; fake quant precision only")
(HERE/"data/summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
fig,ax=plt.subplots(figsize=(9,4));x=np.arange(10);ax.bar(x-.18,[r["bf16_successes"]/r["episodes"] for r in rows],.36,label="BF16");ax.bar(x+.18,[r["w4_successes"]/r["episodes"] for r in rows],.36,label="AWQ W4A16");ax.set_xticks(x);ax.set_xlabel("Spatial task ID");ax.set_ylabel("Success rate");ax.set_ylim(0,1.05);ax.legend();ax.set_title(str(len(pairs))+" matched episodes per variant; fixed profile and seeds");fig.tight_layout()
for ext in ("png","svg"):fig.savefig(HERE/"figures"/("01_per_task."+ext),dpi=160)
plt.close(fig)
fig,ax=plt.subplots(figsize=(12,4));matrix=np.array([[int(r["w4_success"])-int(r["bf16_success"]) for r in pairs if r["task_id"]==t] for t in range(10)]);ax.imshow(matrix,cmap="coolwarm",vmin=-1,vmax=1,aspect="auto");ax.set_yticks(range(10));ax.set_ylabel("Spatial task ID");inds=result["covered_initial_indices"];ax.set_xticks(range(len(inds)));ax.set_xticklabels(inds);ax.set_xlabel("Official initial state index");ax.set_title("W4 - BF16 outcome: blue BF16 only; white same; red W4 only");fig.tight_layout()
for ext in ("png","svg"):fig.savefig(HERE/"figures"/("02_paired_outcomes."+ext),dpi=160)
plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4));labels=["Both succeed","BF16 only","W4 only","Both fail"];counts=[result[k] for k in ("both_success","bf16_only","w4_only","both_failure")];bars=ax.bar(labels,counts,color=["#648A64","#4B71A5","#C46B63","#888888"])
for bar,value in zip(bars,counts):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.5,str(value),ha="center",va="bottom")
ax.set_ylabel("Paired episodes");ax.set_ylim(0,max(counts)*1.15+1);ax.set_title("Matched BF16 / AWQ W4 outcomes; no equivalence claim");fig.tight_layout()
for ext in ("png","svg"):fig.savefig(HERE/"figures"/("03_paired_counts."+ext),dpi=160)
plt.close(fig)
print(json.dumps(result,indent=2))
