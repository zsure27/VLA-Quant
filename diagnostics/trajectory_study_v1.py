"""Offline analysis + scientific figures. --demo is visibly synthetic, never SR.

python -m diagnostics.trajectory_study_v1 --demo --output outputs/trajectory-demo
python -m diagnostics.trajectory_study_v1 --manifest PRIVATE/pairs.json --output PRIVATE/analysis
Manifest: {data_kind, thresholds:{translation_m,rotation_deg,persistence},
           pairs:[{candidate_dir,reference_dir}]}; paths relative to manifest parent.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from qvla.trajectory_diagnostics_v1 import compare_episode, rotation_difference, sustained_windows, rolling_spread


SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def demo_rows():
    """Different planted mechanisms incl. gradual/no onset; not benchmark results."""
    rng = np.random.RandomState(7)
    out = []
    for si, suite in enumerate(SUITES):
        for task in range(10):
            for reset in range(3):
                n = [120, 136, 144, 208][si] + 8*(task % 3)
                t = np.arange(n)
                for model, strength in [("AWQ-W4A16", .5), ("AWQ-W2A16", 1.)]:
                    onset = int(n*(.30+.035*(task % 5)))
                    mode = task % 4
                    envelope = np.maximum(0, t-onset)/max(1,n-onset)
                    if mode == 1:
                        envelope = t/n  # Gradual, no planted abrupt breakpoint.
                    if mode == 2:
                        envelope = np.zeros(n)  # Null example.
                    e = rng.normal(0,.0015,(n,3))
                    e[:,0] += strength*.045*envelope
                    e += rng.normal(0,.008*strength,(n,3))*(envelope[:,None]>.1)
                    rot = rng.normal(0,.001,(n,3))
                    rot[:,2] += strength*.22*envelope
                    if mode == 3:
                        rot[:,2] *= np.sin(t/8)  # Cancellation, not persistent signed drift.
                    tn = np.linalg.norm(e,axis=1)
                    rn = np.linalg.norm(rot,axis=1)*180/np.pi
                    thresholds = {"translation_m": .02, "rotation_deg": 5., "persistence": 3,
                                  "source": "DEMO_ONLY_not_calibrated"}
                    out.append(dict(data_kind="synthetic_demo", suite=suite, task_id=task,
                                    reset_index=reset, candidate=model, reference="BF16",
                                    candidate_status="synthetic_no_outcome", reference_status="synthetic_no_outcome",
                                    common_steps=n, candidate_steps=n, reference_steps=n, censored=False,
                                    translation_vector_m=e.tolist(), rotation_vector_rad_world=rot.tolist(),
                                    translation_norm_m=tn.tolist(), rotation_norm_deg=rn.tolist(),
                                    local_translation_spread_m=rolling_spread(e), thresholds=thresholds,
                                    translation_windows=sustained_windows(tn,.02,3),
                                    rotation_windows=sustained_windows(rn,5.,3)))
    return out


def figures(rows, output, demo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    title = "SYNTHETIC DEMO - NOT EXPERIMENT RESULTS" if demo else "MEASURED | matched-reset descriptive differences"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    files = []

    def save(fig, name):
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0,0,1,.95))
        for suffix in ("png", "pdf"):
            fig.savefig(str(output/(name+"."+suffix)), dpi=145)
        files.append(name+".png")
        plt.close(fig)

    # Deterministic selection, no cherry-picking the largest error.
    chosen = sorted(rows,key=lambda r:(r["suite"],r["task_id"],r["reset_index"],r["candidate"]))[0]
    for r in rows:
        if r["suite"] == "libero_spatial" and r["task_id"] == 0 and r["reset_index"] == 0 and "W2" in r["candidate"]:
            chosen = r; break
    n = chosen["common_steps"]
    t = np.arange(n)
    fig, axes = plt.subplots(3,1,figsize=(10,8),sharex=True)
    for ax, key, label, threshold in zip(axes,
            ("translation_norm_m","rotation_norm_deg","local_translation_spread_m"),
            ("Pose difference (mm)","SO(3) pose difference (deg)","Local temporal scatter (mm)"),
            (chosen["thresholds"]["translation_m"]*1000,chosen["thresholds"]["rotation_deg"],None)):
        factor = 1 if "deg" in label else 1000
        ax.plot(t,np.array(chosen[key])*factor,label=chosen["candidate"])
        if threshold is not None:
            ax.axhline(threshold,color="gray",ls="--",label="Frozen threshold" if not demo else "Illustrative threshold")
        ax.set_ylabel(label); ax.grid(alpha=.2); ax.legend(loc="upper left")
    for key,color in [("translation_windows","tab:blue"),("rotation_windows","tab:orange")]:
        for w in chosen[key][:1]:
            for ax in axes:
                ax.axvline(w["confirmed_index"],color=color,ls=":")
    axes[-1].set_xlabel("Control step (settling excluded); dotted lines = alarm confirmation")
    save(fig,"01-step-traces")

    # Episode-level summaries: steps do NOT become independent samples.
    fig, axes = plt.subplots(1,2,figsize=(11,4.5))
    models = sorted(set(r["candidate"] for r in rows))
    for ax,key,label in [(axes[0],"translation_norm_m","Episode p95 pose difference (mm)"),
                         (axes[1],"rotation_norm_deg","Episode p95 SO(3) difference (deg)")]:
        data,labels=[],[]
        for suite in SUITES:
            for model in models:
                selected=[r for r in rows if r["suite"]==suite and r["candidate"]==model]
                if selected:
                    data.append([np.percentile(r[key],95)*(1000 if key.endswith("_m") else 1) for r in selected])
                    labels.append(suite.replace("libero_","")+"\n"+model.replace("AWQ-",""))
        ax.boxplot(data,labels=labels,showfliers=True)
        ax.set_ylabel(label); ax.tick_params(axis="x",labelrotation=40)
    save(fig,"02-suite-distributions")

    fig, axes = plt.subplots(1,2,figsize=(11,5))
    for ax,key,label in [(axes[0],"translation_windows","Translation alarm: first confirmed fraction"),
                         (axes[1],"rotation_windows","Rotation alarm: first confirmed fraction")]:
        grid = np.full((40,len(models)),-1.)
        for si,suite in enumerate(SUITES):
            for task in range(10):
                for mi,model in enumerate(models):
                    cell=[r for r in rows if r["suite"]==suite and r["task_id"]==task and r["candidate"]==model]
                    values=[r[key][0]["confirmed_index"]/max(1,r["common_steps"]-1) for r in cell if r[key]]
                    if cell: grid[si*10+task,mi]=np.nan
                    if values: grid[si*10+task,mi]=np.median(values)
        cmap=plt.cm.viridis.copy() if hasattr(plt.cm.viridis,"copy") else plt.cm.viridis
        cmap.set_bad("lightgray")
        cmap.set_under("white")
        im=ax.imshow(np.ma.masked_invalid(grid),aspect="auto",vmin=0,vmax=1,cmap=cmap)
        ax.set_xticks(range(len(models))); ax.set_xticklabels(models,rotation=20)
        ax.set_yticks([4.5,14.5,24.5,34.5]); ax.set_yticklabels(["Spatial 0-9","Object 0-9","Goal 0-9","Long 0-9"])
        ax.set_title(label); fig.colorbar(im,ax=ax)
    save(fig,"03-task-alarm-map")

    fig, axes=plt.subplots(1,2,figsize=(11,4.7))
    vectors=np.asarray(chosen["translation_vector_m"])*1000
    axes[0].plot(vectors[:,0],vectors[:,1],lw=1)
    axes[0].scatter(vectors[0,0],vectors[0,1],marker="o",label="Start")
    axes[0].scatter(vectors[-1,0],vectors[-1,1],marker="x",label="End")
    axes[0].set(xlabel="World dx (mm)",ylabel="World dy (mm)",title="Relative position path, not absolute robot path")
    axes[0].legend()
    rot=np.asarray(chosen["rotation_vector_rad_world"])*180/np.pi
    for j,name in enumerate(("world x","world y","world z")):
        axes[1].plot(t,rot[:,j],label=name)
    axes[1].axhline(0,color="gray",lw=.7)
    axes[1].set(xlabel="Control step",ylabel="Signed SO(3) log component (deg)",title="Direction retained; no sum of angle magnitudes")
    axes[1].legend()
    save(fig,"04-direction-and-drift")

    fig,axes=plt.subplots(2,4,figsize=(13,7),sharex="row")
    for si,suite in enumerate(SUITES):
        selected=[r for r in rows if r["suite"]==suite and r["candidate"]==chosen["candidate"]]
        for row,key,factor,label in [(0,"translation_norm_m",1000,"Position diff (mm)"),(1,"rotation_norm_deg",1,"Rotation diff (deg)")]:
            ax=axes[row,si]
            for q,labelq in [(0,"first third"),(1,"middle third"),(2,"last third")]:
                # Equal episode weight: one median per third per episode, NOT step pooling.
                values=[np.median(np.array_split(np.asarray(r[key]),3)[q])*factor for r in selected]
                if values: ax.hist(values,bins=8,alpha=.45,label=labelq)
            ax.set_title(suite.replace("libero_","")); ax.set_xlabel(label)
            ax.set_ylabel("Episodes")
    axes[0,0].legend(fontsize=8)
    save(fig,"05-distribution-evolution")

    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,key,label in [(axes[0],"translation_windows","Translation alarm"),(axes[1],"rotation_windows","Rotation alarm")]:
        for model in models:
            selected=[r for r in rows if r["candidate"]==model]
            xs=np.linspace(0,1,31)
            # Observed-domain fraction, not survival analysis: denominators shown explicitly.
            ys=[sum(bool(r[key]) and r[key][0]["confirmed_index"]<=x*max(1,r["common_steps"]-1) for r in selected)/len(selected) for x in xs]
            ax.plot(xs,ys,label=model+" (n="+str(len(selected))+")")
        ax.set(xlabel="Within common observed horizon (descriptive only)",ylabel="Fraction with confirmed alarm",title=label,ylim=(0,1))
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    save(fig,"06-alarm-coverage")
    return files


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo",action="store_true")
    mode.add_argument("--manifest",type=Path)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh output directory; preserve prior analysis")
    if args.demo:
        rows=demo_rows()
    else:
        manifest=json.loads(args.manifest.read_text(encoding="utf-8"))
        if manifest["data_kind"] != "measured" or not manifest.get("thresholds",{}).get("calibration_sha256"):
            raise ValueError("Measured manifest needs frozen calibration threshold SHA")
        calibration=args.manifest.parent/manifest["thresholds"]["calibration_file"]
        if hashlib.sha256(calibration.read_bytes()).hexdigest()!=manifest["thresholds"]["calibration_sha256"]:
            raise ValueError("Threshold calibration content changed")
        frozen=json.loads(calibration.read_text(encoding="utf-8"))
        if frozen.get("data_kind")!="measured" or any(frozen[k]!=manifest["thresholds"][k] for k in ("translation_m","rotation_deg","persistence")):
            raise ValueError("Uncalibrated or altered thresholds")
        rows=[compare_episode(args.manifest.parent/r["candidate_dir"],args.manifest.parent/r["reference_dir"],manifest["thresholds"]) for r in manifest["pairs"]]
        if any(r["data_kind"]!="measured" for r in rows):
            raise ValueError("Synthetic data cannot enter measured analysis")
        keys=[(r["suite"],r["task_id"],r["reset_index"],r["candidate"]) for r in rows]
        if len(set(keys))!=len(keys): raise ValueError("Duplicate candidate episode pair")
    args.output.mkdir(parents=True)
    files=figures(rows,args.output,args.demo)
    # Raw vectors stay in private analysis; public summaries must be explicitly reviewed.
    coverage={suite:{m:sum(r["suite"]==suite and r["candidate"]==m for r in rows)
                     for m in sorted(set(r["candidate"] for r in rows))} for suite in SUITES}
    cells=[]
    for suite in SUITES:
        for task in range(10):
            for model in sorted(set(r["candidate"] for r in rows)):
                cell=[r for r in rows if r["suite"]==suite and r["task_id"]==task and r["candidate"]==model]
                cells.append(dict(suite=suite,task_id=task,candidate=model,observed_pairs=len(cell),
                    translation_alarm_pairs=sum(bool(r["translation_windows"]) for r in cell),
                    rotation_alarm_pairs=sum(bool(r["rotation_windows"]) for r in cell),
                    censored_pairs=sum(r["censored"] for r in cell)))
    (args.output/"analysis.json").write_text(json.dumps({"data_kind":"synthetic_demo" if args.demo else "measured",
        "coverage_pairs_by_suite":coverage,"coverage_task_cells":cells,
        "censored_pairs":sum(r["censored"] for r in rows),"comparisons":rows},allow_nan=False),encoding="utf-8")
    heading="SIMULATED PREVIEW / NOT MEASURED" if args.demo else "MEASURED DESCRIPTIVE ANALYSIS"
    images="".join('<figure><img loading="lazy" src="'+f+'" alt="'+f+'"><figcaption>'+f+'</figcaption></figure>' for f in files)
    html='<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>VLA trajectory diagnostics</title><style>body{font:16px system-ui;max-width:1200px;margin:auto;padding:16px}img{width:100%;height:auto}figure{margin:24px 0}header{position:sticky;top:0;background:#fff3cf;color:#422b00;padding:12px}a{margin-right:12px}</style><header>'+heading+'</header><p>Six views; no experimental success-rate claim. Gray heatmap cells = no detected alarm; white = missing pair coverage. Colored cells summarize only detected cases: read per-cell denominators in analysis.json. Threshold crossings are not causal failure proofs.</p>'+images
    (args.output/"index.html").write_text(html,encoding="utf-8")
    print(json.dumps({"data_kind":"synthetic_demo" if args.demo else "measured","pairs":len(rows),"figures":files}))


if __name__ == "__main__": main()
