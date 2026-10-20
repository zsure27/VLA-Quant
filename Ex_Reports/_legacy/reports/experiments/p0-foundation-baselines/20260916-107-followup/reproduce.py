"""Plot measured follow-up data; do not fabricate missing or partial tests."""
import csv
import hashlib
import json
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RAW = REPO / "results/experiments/p0-foundation-baselines/20260916-107-followup"
OLD = REPO / "results/experiments/p0-foundation-baselines/20260916-107-diagnostics"
DATA = HERE / "data"
FIG = HERE / "figures"
DATA.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"svg.hashsalt": "vla107followup", "font.size": 9})

def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))

def write(name, rows):
    with (DATA / name).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)

def save(fig, name, note):
    fig.text(.02, .015, note, fontsize=8)
    fig.tight_layout(rect=[0, .08, 1, 1])
    fig.savefig(FIG / (name+".png"), dpi=180)
    fig.savefig(FIG / (name+".svg"), metadata={"Date": None})
    p = FIG / (name+".svg")
    p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n"))
    plt.close(fig)

paths = {
    "W2 baseline": OLD / "baseline/metrics.json",
    "W4 bits / W2 coordinates": RAW / "awq-family-precision-20260916-231456-1371/all/metrics.json",
    "W4 searched profile": OLD / "awq-language-stage-rescue-a-20260916-185313-2064/stage-08-15/metrics.json",
}
records = {k: read(p) for k,p in paths.items()}
source_paths=set(paths.values())
names = list(records)
samples = [[r["sample"] for r in records[n]] for n in names]
assert samples[0] == samples[1] == samples[2] and len(samples[0]) == 32
errors = {n: np.array([r["normalized_action"]["mse"] for r in records[n]]) for n in names}
rows = [{"configuration": n, "frames": 32, "mean_action_mse": float(errors[n].mean()),
         "gripper_disagreement_steps": round(sum(r["raw_gripper_disagreement"]*8 for r in records[n])),
         "evidence_class": "measured_development_frames"} for n in names]
write("awq_summary.csv", rows)
write("awq_paired_frames.csv", [{"sample": samples[0][i], **{n:float(errors[n][i]) for n in names},
    "fixed_minus_searched": float(errors[names[1]][i]-errors[names[2]][i])} for i in range(32)])
fig, ax = plt.subplots(1,2,figsize=(11,4))
ax[0].bar(range(3), [r["mean_action_mse"] for r in rows], color=["#999999", "#3a86ff", "#2a9d8f"])
ax[0].set_xticks(range(3), ["W2 baseline", "W4 bits\nW2 coordinates", "W4 searched\nprofile"])
ax[0].set_ylabel("Mean normalized action MSE vs BF16 teacher")
ax[1].scatter(errors[names[2]], errors[names[1]])
upper=max(errors[names[1]].max(), errors[names[2]].max())
ax[1].plot([0,upper],[0,upper], "k--", linewidth=1)
ax[1].set_xlabel("W4 searched profile: frame MSE")
ax[1].set_ylabel("W4 bits / W2 coordinates: frame MSE")
save(fig, "01_awq_fixed_coordinates", "32 paired development inputs; vision BF16; layers 8-15 rescued; fake quant, no PEFT.")

sq = next(iter(RAW.glob("sq-controls-recheck-*/primary-only/metrics.json")), None)
if sq:
    branch = sq.parents[1]
    specs = {"Vision both (previous)": OLD / "sq-controls-recheck-20260916-192316-6105/no-language/metrics.json",
             "DINO only": branch / "primary-only/metrics.json", "SigLIP only": branch / "fused-only/metrics.json"}
    rr = {n: read(p) for n,p in specs.items()}
    source_paths.update(specs.values())
    ss = [[r["sample"] for r in records] for records in rr.values()]
    assert len(ss[0]) == 8 and all(s == ss[0] for s in ss)
    sr = [{"configuration": n, "frames": 8,
           "mean_action_mse": float(np.mean([r["normalized_action"]["mse"] for r in rs])),
           "max_action_mse": max(r["normalized_action"]["mse"] for r in rs),
           "gripper_disagreement_steps": round(sum(r["raw_gripper_disagreement"]*8 for r in rs)),
           "within_original_1e4_gate": max(r["normalized_action"]["mse"] for r in rs)<=1e-4} for n,rs in rr.items()]
    write("sq_branch_summary.csv", sr)
    write("sq_branch_frames.csv", [{"sample": ss[0][i], **{n:rs[i]["normalized_action"]["mse"] for n,rs in rr.items()}} for i in range(8)])
    fig, ax=plt.subplots(figsize=(8,4))
    for n,rs in rr.items(): ax.plot(range(8), [r["normalized_action"]["mse"] for r in rs], "o-", label=n)
    ax.axhline(1e-4, color="k", linestyle="--", label="Original engineering gate")
    ax.set_xlabel("Paired control input index (samples 64-71)"); ax.set_ylabel("Action MSE vs BF16 teacher")
    ax.legend(fontsize=8)
    save(fig, "02_sq_vision_branches", "W16A16: smoothing only; language unsmoothed. Curves cannot be added as independent causes.")

lowrank=next(iter(RAW.glob("awq-low-rank-residual-*")),None)
if lowrank and (lowrank/"complete.json").exists():
    summaries=[]; perframe=[]
    for rank in (4,8):
        rs=read(lowrank/f"rank-{rank}/metrics.json")
        scope=read(lowrank/f"rank-{rank}/scope.json")
        assert [r["sample"] for r in rs]==samples[0]
        assert scope["residual_training_steps"]==0 and len(scope["low_rank_residual"])==56
        summaries.append({"rank":rank,"frames":32,"mean_action_mse":float(np.mean([r["normalized_action"]["mse"] for r in rs])),
            "gripper_disagreement_steps":round(sum(r["raw_gripper_disagreement"]*8 for r in rs)),
            "adapter_parameters":scope["residual_adapter_parameters"],"training_steps":0,
            "mean_unexplained_weight_residual":float(np.mean([r["residual_frobenius_unexplained_fraction"] for r in scope["low_rank_residual"].values()]))})
        perframe.extend({"rank":rank,"sample":r["sample"],"action_mse":r["normalized_action"]["mse"]} for r in rs)
    write("low_rank_summary.csv",summaries); write("low_rank_frames.csv",perframe)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar(["W2 baseline","Residual rank 4","Residual rank 8","Stage 8-15 W4"],
        [rows[0]["mean_action_mse"],summaries[0]["mean_action_mse"],summaries[1]["mean_action_mse"],rows[1]["mean_action_mse"]])
    ax.set_ylabel("Mean action MSE vs BF16 teacher")
    save(fig,"06_low_rank_residual","32 development inputs, layers 8-15, BF16 residual branch; SVD initialization, zero training steps. No closed-loop claim.")

local = next(iter(RAW.glob("sq-local*/measurement/metrics.json")), None)
if local:
    lr=read(local)
    write("sq_local_groups.csv", lr)
    cases=["fp32", "bf16", "fp32_transform_cast_bf16"]
    group_names=list(dict.fromkeys(r["norm"] for r in lr))
    fig,ax=plt.subplots(figsize=(11,5))
    for case in cases:
        vals=[next(r["relative_mse"] for r in lr if r["case"]==case and r["norm"]==n) for n in group_names]
        ax.plot(range(len(vals)),vals,"o-",label=case)
    ax.set_yscale("log"); ax.set_ylabel("Local output relative MSE")
    ax.set_xticks(range(len(group_names)),[n.replace("vision_backbone.","").replace("featurizer.blocks.","DINO/").replace("fused_DINO/","SigLIP/") for n in group_names],rotation=60,ha="right")
    ax.legend()
    save(fig,"03_sq_local_equivalence","Same captured input per group; sample 65, first camera, 8 tokens. Local errors do not prove end-to-end behavior.")

def rollout(path, expected):
    ff=list(path.glob("EVAL-*.txt"))
    assert len(ff)==1 and (path/"exit-code.txt").read_text().strip()=="0"
    text=ff[0].read_text(encoding="utf-8")
    source_paths.add(ff[0])
    mm=[json.loads(x) for x in re.findall(r"^EPISODE_MANIFEST (.+)$",text,re.M)]
    ss=[x=="True" for x in re.findall(r"^Success: (True|False)$",text,re.M)]
    assert len(mm)==len(ss)==expected and "Episode error:" not in text
    return mm,ss

ten=RAW/"language-stage-rescue10-20260916-232303-2556/stage-08-15"
if ten.exists():
    runs={"Fixed W2 coordinates, stage 8-15 W4":ten,
          "Searched W4 profile, stage 8-15":OLD/"language-stage-rescue10-20260916-191437-4331/stage-08-15"}
    outcomes={}; paired=None; cr=[]
    for label,path in runs.items():
        mm,ss=rollout(path,10)
        if paired is None: paired=mm
        assert mm==paired
        outcomes[label]=ss
        cr.extend({"configuration":label,**m,"success":s} for m,s in zip(mm,ss))
    write("closed_loop_10.csv",cr)
    fig,ax=plt.subplots(figsize=(10,3))
    ax.imshow(list(outcomes.values()),aspect="auto",vmin=0,vmax=1,cmap="RdYlGn")
    ax.set_yticks(range(2),[f"{n}: {sum(ss)}/10" for n,ss in outcomes.items()])
    ax.set_xticks(range(10)); ax.set_xlabel("Spatial task ID; initial state 0")
    save(fig,"04_fixed_coordinates_closed_loop","Ten development states; paired manifests exactly equal. Small exploratory sample, no statistical superiority claim.")

fifty=next(iter(RAW.glob("language-stage-rescue50-*/stage-08-23")),None)
if fifty and (fifty/"exit-code.txt").exists():
    mm,ss=rollout(fifty,50)
    refs={n:rollout(REPO/"results/experiments/p0-foundation-baselines/20260916-awq-baseline-gate"/n,50) for n in ("bf16","w4")}
    assert all(m==mm for m,s in refs.values())
    results={"BF16":refs["bf16"][1],"W4 all":refs["w4"][1],"W2 + stage 8-23 W4":ss}
    write("closed_loop_50.csv",[{"configuration":n,**m,"success":v} for n,vs in results.items() for m,v in zip(mm,vs)])
    write("closed_loop_50_summary.csv",[{"configuration":n,"successes":sum(vs),"episodes":50,
        "new_candidate_init_1_to_4_successes":sum(v for m,v in zip(mm,vs) if m["init_state_index"]>0),"new_candidate_init_1_to_4_episodes":40} for n,vs in results.items()])
    fig,ax=plt.subplots(figsize=(12,3.5))
    ax.imshow(list(results.values()),aspect="auto",vmin=0,vmax=1,cmap="RdYlGn")
    ax.set_yticks(range(3),[f"{n}: {sum(vs)}/50" for n,vs in results.items()])
    ax.set_xticks([5*i+2 for i in range(10)],range(10)); ax.set_xlabel("Spatial task ID; five initial states per task (0-4)")
    save(fig,"05_stage_08_23_closed_loop50","Exact paired manifests against previous BF16/W4 runs. Mixed precision, vision BF16; no PEFT or packed kernel.")

for prefix,label in (("sq-fp32-pairs-*","FP32 norm/Linear pairs"),("sq-fp32-vision-*","Whole vision FP32")):
    directory=next(iter(RAW.glob(prefix)),None)
    if directory and (directory/"complete.json").exists():
        sr=[]
        for case,ref in (("fp32-repeat","original BF16"),("fp32-smooth",label+" unsmoothed")):
            rs=read(directory/case/"metrics.json")
            assert len(rs)==8
            scope=read(directory/case/"scope.json")
            sr.append({"case":case,"reference":ref,"frames":8,
                "mean_action_mse":float(np.mean([r["normalized_action"]["mse"] for r in rs])),
                "max_action_mse":max(r["normalized_action"]["mse"] for r in rs),
                "gripper_disagreement_steps":round(sum(r["raw_gripper_disagreement"]*8 for r in rs)),
                "smoothing_groups":scope["smoothing_groups"],
                "within_original_1e4_gate":max(r["normalized_action"]["mse"] for r in rs)<=1e-4})
        tag="pairs" if "pairs" in prefix else "vision"
        write(f"sq_fp32_{tag}_summary.csv",sr)
        write(f"sq_fp32_{tag}_frames.csv",[{"case":case,"sample":r["sample"],"action_mse":r["normalized_action"]["mse"]} for case in ("fp32-repeat","fp32-smooth") for r in read(directory/case/"metrics.json")])
        fig,ax=plt.subplots(figsize=(8,4))
        for case in ("fp32-repeat","fp32-smooth"):
            rs=read(directory/case/"metrics.json")
            ax.plot(range(8),[r["normalized_action"]["mse"] for r in rs],"o-",label=case)
        ax.axhline(1e-4,color="k",linestyle="--",label="Original engineering gate")
        ax.set_xlabel("Control input index (samples64-71)"); ax.set_ylabel("Action MSE"); ax.legend()
        save(fig,f"07_sq_fp32_{tag}",f"{label}. Repeat vs BF16; smooth vs its own FP32 repeat. Different references; not low-bit/PEFT results.")

weighted=next(iter(RAW.glob("awq-input-diag-residual-*")),None)
if weighted and (weighted/"complete.json").exists():
    wr=[]; wf=[]
    for case in ("all","action"):
        rs=read(weighted/f"rank-{case}/metrics.json")
        scope=read(weighted/f"rank-{case}/scope.json")
        assert len(rs)==32 and [r["sample"] for r in rs]==samples[0]
        assert len(scope["residual_calibration_manifest"])==8 and scope["residual_training_steps"]==0
        wr.append({"initialization":case+" token diagonal RMS","rank":8,"frames":32,"calibration_frames":8,
            "adapter_parameters":scope["residual_adapter_parameters"],"training_steps":0,
            "mean_action_mse":float(np.mean([r["normalized_action"]["mse"] for r in rs])),
            "gripper_disagreement_steps":round(sum(r["raw_gripper_disagreement"]*8 for r in rs)),
            "mean_unexplained_weight_residual":float(np.mean([r["residual_frobenius_unexplained_fraction"] for r in scope["low_rank_residual"].values()])),
            "mean_unexplained_input_diagonal_residual":float(np.mean([r["input_diagonal_unexplained_fraction"] for r in scope["low_rank_residual"].values()]))})
        wf.extend({"initialization":case,"sample":r["sample"],"action_mse":r["normalized_action"]["mse"]} for r in rs)
    write("input_diag_summary.csv",wr); write("input_diag_frames.csv",wf)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar(["Weight SVD rank8","All-token RMS rank8","Action-token RMS rank8"],
        [float(next(r for r in summaries if r["rank"]==8)["mean_action_mse"]),wr[0]["mean_action_mse"],wr[1]["mean_action_mse"]])
    ax.set_ylabel("Mean action MSE vs BF16")
    save(fig,"08_input_diag_residual","Same rank/parameter budget; 8 calibration, 32 development inputs; diagonal covariance approximation, zero training.")

files = list(RAW.rglob("*"))+list(DATA.rglob("*"))+list(FIG.rglob("*"))+list(source_paths)+list((HERE/"backup").rglob("*"))+[Path(__file__),HERE/"README_CN.md",HERE/"closure.json"]
manifest = {str(p.relative_to(REPO)).replace("\\", "/"): hashlib.sha256(p.read_bytes().replace(b"\r\n",b"\n") if p.suffix in (".json", ".py", ".svg", ".csv", ".txt", ".log", ".md", ".sh") else p.read_bytes()).hexdigest() for p in files if p.is_file()}
(HERE / "manifest.json").write_text(json.dumps({"normalization": "LF for text; bytes otherwise", "files":manifest}, indent=2)+"\n", encoding="utf-8")
print(json.dumps(rows, indent=2))
