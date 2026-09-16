import json,csv,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
RAW=REPO/"results/107-response-svd-20260917/awq-response-svd-20260917-003557-1297"
OLD=REPO/"results/107-followup-20260916/awq-input-diag-residual-20260916-235145-8370"
def read(p): return json.loads(p.read_text(encoding="utf-8"))
rows=[];frames=[];groups=[];samples=None;calibration=None
cases=[("Action RMS rank8",OLD/"rank-action"),("Response SVD rank8",RAW/"rank-8"),("Response SVD rank16",RAW/"rank-16")]
controls=list((REPO/"results/107-response-svd-20260917").glob("awq-response-svd-rms16-*/rank-16"))
if controls: cases.append(("Action RMS rank16",controls[0]))
for label,p in cases:
 m=read(p/"metrics.json");s=read(p/"scope.json")
 assert len(m)==32 and s["residual_training_steps"]==0 and len(s["residual_calibration_manifest"])==8
 ids=[r["sample"] for r in m]
 if calibration is None: calibration=s["residual_calibration_manifest"]
 else: assert s["residual_calibration_manifest"]==calibration
 if samples is None: samples=ids
 else: assert ids==samples
 rows.append({"case":label,"frames":32,"calibration_frames":8,"parameters":s["residual_adapter_parameters"],"training_steps":0,"mean_action_mse":float(np.mean([r["normalized_action"]["mse"] for r in m])),"gripper_disagreement_steps":round(sum(r["raw_gripper_disagreement"]*8 for r in m)),"mean_unexplained_weight_residual":float(np.mean([v["residual_frobenius_unexplained_fraction"] for v in s["low_rank_residual"].values()])),"mean_unexplained_calibration_response":None if label.startswith("Action") else float(np.mean([v["response_unexplained_fraction"] for v in s["low_rank_residual"].values()]))})
 frames.extend({"case":label,"sample":r["sample"],"action_mse":r["normalized_action"]["mse"],"gripper_disagreement":r["raw_gripper_disagreement"]} for r in m)
 if not label.startswith("Action"):
  assert s["residual_response_svd"]
  groups.extend(dict(case=label,module=n,**v) for n,v in s["low_rank_residual"].items())
(HERE/"data").mkdir(exist_ok=True);(HERE/"figures").mkdir(exist_ok=True)
for name,values in [("summary",rows),("frames",frames),("module_geometry",groups)]:
 with (HERE/"data"/(name+".csv")).open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
(HERE/"data/summary.json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")
fig,ax=plt.subplots(figsize=(11,4.5));ax.bar([r["case"] for r in rows],[r["mean_action_mse"] for r in rows]);ax.axhline(.06972660918836482,color="k",ls="--",label="W2 baseline");ax.set_ylabel("Mean action MSE vs BF16");ax.set_title("8 calibration / 32 reused development inputs; zero training");ax.legend();fig.tight_layout()
for ext in ("png","svg"):fig.savefig(HERE/"figures"/("01_action_mse."+ext),dpi=160)
plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4.5))
for label in [r["case"] for r in rows]:ax.plot(range(32),[r["action_mse"] for r in frames if r["case"]==label],label=label)
ax.set_xlabel("Development input index (samples1000-1031)");ax.set_ylabel("Action MSE vs BF16");ax.legend();fig.tight_layout()
for ext in ("png","svg"):fig.savefig(HERE/"figures"/("02_paired_frames."+ext),dpi=160)
plt.close(fig)
print(json.dumps(rows,indent=2))
