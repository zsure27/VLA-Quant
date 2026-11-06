"""PPT-inspired W/A diagnostics. Synthetic preview is NEVER benchmark evidence.

--bundle consumes a privately audited measured JSON bundle; schema is illustrated
by --demo output. Measured bundle SHA/lineage and all four W/A input identities
must be verified separately at capture/replay time. No GPU imports or new rollout.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
import numpy as np
from qvla.trajectory_diagnostics_v1 import finite, rotation_difference
from qvla.trajectory_diagnostics_v2 import (factorial_errors,boundary_projection,
    event_aligned,activation_summary,demo_symmetric_qdq)


def synthetic_bundle():
    rng=np.random.RandomState(20261106)
    n,h=160,8;t=np.arange(n);q=np.arange(n//h);j=np.arange(h)
    pos=np.c_[.22+.28*t/n,.08*np.sin(2.6*t/n),.12+.15*np.sin(np.pi*t/n)**2]
    base=np.zeros((len(q),h,7))
    for k in range(6):base[...,k]=.15*np.sin(q[:,None]*.12+j[None,:]*.18+k)
    base[...,-1]=(q[:,None]>=6).astype(float)
    w=np.zeros_like(base);a=np.zeros_like(base);inter=np.zeros_like(base)
    w[...,0]=.025*np.sin(q[:,None]/4)+j[None,:]*.002
    w[...,3]=.005*q[:,None]
    envelope=(q[:,None]>=7)*(j[None,:]+1)/h
    a[...,1]=.11*envelope;a[...,2]=-.07*envelope
    inter[...,1]=.08*envelope*np.sin(q[:,None]*.25)
    conditions=[]
    settings=[("BF16",0.,0.),("AWQ-W4A16",.35,0.),("AWQ-W2A16",1.4,0.),
              ("SQ-W4A16",.45,0.),("SQ-W4A8",.45,.4),("SQ-W4A4",.45,1.)]
    for name,ws,ac in settings:
        err=(ws*w+ac*a+ac*inter)
        pose=pos.copy()
        growth=np.maximum(0,t-40)/120
        pose[:,0]+=ws*.025*growth
        pose[:,1]+=ac*.045*(t>=56)*(1-np.exp(-np.maximum(0,t-56)/8))
        pose[:,2]+=ws*.004*np.sin(t/6)+ac*.012*growth
        angle=ws*.14*growth+ac*.04*np.maximum(0,t-80)/80
        quat=np.c_[np.zeros(n),np.zeros(n),np.sin(angle/2),np.cos(angle/2)]
        commands=np.zeros((n,7));commands[:,-1]=np.where(t<48,-1,1)
        conditions.append(dict(id=name,positions_m=pose.tolist(),quaternions_xyzw=quat.tolist(),
            controller_commands=commands.tolist(),raw_chunks=(base+err).tolist(),
            teacher_chunks_same_observation=base.tolist(),query_control_steps=(q*h).tolist(),
            input_sha256="synthetic-shared-input",coordinate_id="synthetic-fixed-coordinate",
            executed_mask=np.ones((len(q),h),dtype=bool).tolist()))
    events=[]
    for k in range(12):
        event=42+k%5
        x=np.arange(110)
        error=np.abs(rng.normal(0,.001,len(x)))+.025/(1+np.exp(-(x-event)/3))
        events.append(dict(error_m=error.tolist(),event_index=None if k==11 else event))
    activations=np.r_[rng.normal(0,1,6000),rng.normal(0,10,30)]
    return dict(data_kind="synthetic_demo",scenario="Toy illustration, NOT LIBERO or measured model output",focus_control_step=56,
                command_scales=[1]*6,conditions=conditions,
                factorial=dict(fp=base.tolist(),weight_only=(base+w).tolist(),activation_only=(base+a).tolist(),
                               joint=(base+w+a+inter).tolist(),input_hashes=["synthetic-shared-input"]*4,
                               coordinate_ids=["synthetic-fixed-coordinate"]*4),
                event_traces=events,activation_values=activations.tolist(),calibration_absmax=4.)


def validate_bundle(bundle):
    if bundle.get("data_kind") not in ("measured","synthetic_demo"):raise ValueError("Explicit data_kind required")
    ids=[c["id"] for c in bundle["conditions"]]
    if len(set(ids))!=len(ids) or "BF16" not in ids:raise ValueError("Unique conditions including BF16 required")
    scales=finite(bundle["command_scales"],(6,))
    if np.any(scales<=0):raise ValueError("Positive fixed command scales required")
    if type(bundle.get("focus_control_step")) is not int or bundle["focus_control_step"]<0:
        raise ValueError("Explicit audited focus control step required")
    reference=next(c for c in bundle["conditions"] if c["id"]=="BF16")
    if bundle["focus_control_step"]>=len(reference["positions_m"]):raise ValueError("Focus outside reference observation horizon")
    for c in bundle["conditions"]:
        pos=finite(c["positions_m"])
        if pos.ndim!=2 or pos.shape[1]!=3:raise ValueError("Expected Tx3 positions")
        qs=finite(c["quaternions_xyzw"])
        if qs.shape!=(len(pos),4):raise ValueError("Pose length mismatch")
        for quat in qs:rotation_difference(quat,[0,0,0,1])
        commands=finite(c["controller_commands"])
        if commands.shape!=(len(pos),7):raise ValueError("Command length mismatch")
        s,t=finite(c["raw_chunks"]),finite(c["teacher_chunks_same_observation"])
        if s.shape!=t.shape or s.ndim!=3 or s.shape[-1]!=7:raise ValueError("Same-observation chunk mismatch")
        mask=np.asarray(c["executed_mask"])
        if mask.shape!=s.shape[:2] or mask.dtype.kind!='b':raise ValueError("Actual executed-slot mask required")
        starts=c["query_control_steps"]
        if len(starts)!=len(s) or any(type(k)is not int or k<0 or k>=len(pos) for k in starts) or starts!=sorted(set(starts)):
            raise ValueError("Query/control-step mapping malformed")
        if not c.get("input_sha256") or not c.get("coordinate_id"):raise ValueError("Query input identity required")
    f=bundle["factorial"]
    return factorial_errors(f["fp"],f["weight_only"],f["activation_only"],f["joint"],f["input_hashes"],f["coordinate_ids"])


def plot_bundle(bundle,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    terms=validate_bundle(bundle)
    demo=bundle["data_kind"]=="synthetic_demo"
    title="SYNTHETIC DEMO - NOT MEASURED VLA RESULTS" if demo else "MEASURED | audited W/A diagnostic bundle"
    cs=bundle["conditions"];ref=next(c for c in cs if c["id"]=="BF16")
    colors={c["id"]:plt.cm.tab10(i) for i,c in enumerate(cs)}
    files=[];metrics={"data_kind":bundle["data_kind"],"scenario":bundle.get("scenario"),"conditions":{}}
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    def save(fig,name):
        fig.suptitle(title,fontsize=13);fig.tight_layout(rect=(0,0,1,.90))
        for ext in ("png","pdf"):fig.savefig(str(output/(name+'.'+ext)),dpi=145)
        plt.close(fig);files.append(name+'.png')

    # 1. True space trajectories, same units, and a deterministic fixed zoom.
    fig=plt.figure(figsize=(12,9));axes=[fig.add_subplot(221,projection="3d"),fig.add_subplot(222),fig.add_subplot(223),fig.add_subplot(224)]
    focus=bundle["focus_control_step"];lo=max(0,focus-16);hi=focus+25
    for c in cs:
        p=np.asarray(c["positions_m"])*1000;color=colors[c["id"]]
        axes[0].plot(p[:,0],p[:,1],p[:,2],label=c["id"],color=color)
        axes[1].plot(p[:,0],p[:,1],color=color)
        axes[2].plot(p[:,0],p[:,2],color=color)
        axes[3].plot(p[lo:hi,0],p[lo:hi,1],color=color)
    axes[0].set(xlabel="x (mm)",ylabel="y (mm)",zlabel="z (mm)",title="End-effector paths; independent closed loops")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=9,loc="upper center",bbox_to_anchor=(.5,.945),ncol=6)
    for ax,label in [(axes[1],"XY projection"),(axes[2],"XZ projection"),(axes[3],"Audited focus window "+str(lo)+":"+str(hi))]:
        ax.set(xlabel="x (mm)",ylabel="z (mm)" if ax==axes[2] else "y (mm)",title=label);ax.grid(alpha=.2)
    save(fig,"07-spatial-path-and-zoom")

    fig,axes=plt.subplots(4,1,figsize=(12,9),sharex=True)
    poses=[]
    for c in cs[1:]:
        n=min(len(c["positions_m"]),len(ref["positions_m"]))
        delta=(np.asarray(c["positions_m"])[:n]-np.asarray(ref["positions_m"])[:n])*1000
        rot=np.array([rotation_difference(a,b) for a,b in zip(c["quaternions_xyzw"][:n],ref["quaternions_xyzw"][:n])])*180/np.pi
        for dim in range(3):axes[dim].plot(np.arange(n),delta[:,dim],label=c["id"],color=colors[c["id"]])
        axes[3].plot(np.arange(n),rot[:,2],label=c["id"],color=colors[c["id"]])
        metrics["conditions"][c["id"]]={"common_pose_steps":n,"pose_horizon_censored":len(c["positions_m"])!=len(ref["positions_m"])}
        poses.append((c,np.linalg.norm(delta,axis=1)))
    for ax,label in zip(axes,["World dx (mm)","World dy (mm)","World dz (mm)","World log-rotation z (deg)"]):
        ax.set_ylabel(label);ax.axhline(0,color="gray",lw=.6);ax.grid(alpha=.2)
    axes[0].legend(ncol=3,fontsize=8);axes[-1].set_xlabel("Control step; pose difference is NOT same-observation action error")
    save(fig,"08-signed-step-components")

    fig,axes=plt.subplots(2,2,figsize=(12,7));scale=np.asarray(bundle["command_scales"])
    for ax,key in zip(axes.flat,["weight","activation","interaction","total"]):
        val=np.linalg.norm(terms[key][...,:6]/scale,axis=-1)
        ax.plot(np.arange(len(val)),np.median(val,axis=1),label="Median across predicted chunk slots")
        ax.fill_between(np.arange(len(val)),np.min(val,axis=1),np.max(val,axis=1),alpha=.15,label="Slot range, not CI")
        ax.set(xlabel="Same-input query index",ylabel="Scaled six-channel command L2",title=key);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    save(fig,"09-weight-activation-factorial")

    selected=[c for c in cs if c["id"].startswith("SQ-W4")]
    if not selected:selected=cs[1:4]
    fig,axes=plt.subplots(1,len(selected),figsize=(max(6,4*len(selected)),6),squeeze=False)
    mats=[]
    for c in selected:
        err=np.linalg.norm((np.asarray(c["raw_chunks"])[...,:6]-np.asarray(c["teacher_chunks_same_observation"])[...,:6])/scale,axis=-1)
        err=np.where(np.asarray(c["executed_mask"]),err,np.nan);mats.append(err)
    cap=max(float(np.nanmax(x)) for x in mats) or 1
    for ax,c,mat in zip(axes[0],selected,mats):
        im=ax.imshow(np.ma.masked_invalid(mat),origin="lower",aspect="auto",vmin=0,vmax=cap,cmap="viridis")
        ax.set(title=c["id"],xlabel="Predicted/executed chunk slot",ylabel="Query index");fig.colorbar(im,ax=ax)
    save(fig,"10-query-by-chunk-slot")

    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for c in selected:
        projection=boundary_projection(c["raw_chunks"],c["teacher_chunks_same_observation"],scale)
        executed=np.asarray(c["executed_mask"])
        valid=projection["valid"] & executed[:,:-1] & executed[:,1:]
        x=projection["signed_parallel"][valid];y=projection["perpendicular"][valid]
        axes[0].scatter(x,y,s=9,alpha=.45,label=c["id"],color=colors[c["id"]])
        angles=np.sort(projection["angle_deg"][valid]);angles=angles[np.isfinite(angles)]
        if len(angles):axes[1].plot(angles,np.arange(1,len(angles)+1)/len(angles),label=c["id"],color=colors[c["id"]])
    axes[0].axvline(0,color="gray",lw=.7)
    axes[0].set(xlabel="Signed parallel command error",ylabel="Perpendicular command error",title="Local action-change direction, not verified mode boundary")
    axes[1].set(xlabel="Signed direction angle (deg, 0-180)",ylabel="Fraction of valid predicted-slot pairs",xlim=(0,180))
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    save(fig,"11-action-change-direction")

    events=bundle.get("event_traces",[])
    if events:
        aligned=event_aligned([r["error_m"] for r in events],[r["event_index"] for r in events])
        fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
        for trace in aligned["values"]:axes[0].plot(aligned["offsets"],trace*1000,color="gray",alpha=.2,lw=.7)
        median=np.nanmedian(aligned["values"],axis=0)*1000
        axes[0].plot(aligned["offsets"],median,label="Median of observed episodes",lw=2)
        axes[0].set_ylabel("Pose difference (mm)");axes[0].legend()
        axes[1].step(aligned["offsets"],aligned["counts"],where="mid")
        axes[1].set_ylim(0,max(1,len(events)))
        axes[1].set(ylabel="Observed episodes",xlabel="Control steps from gripper-close command (not grasp confirmation)",title="No-event episodes retained separately: "+str(aligned["no_event"]))
        for ax in axes:ax.axvline(0,color="gray",ls="--");ax.grid(alpha=.2)
        save(fig,"12-event-aligned-window")
        metrics["event_alignment"]={"no_event":aligned["no_event"],"counts":aligned["counts"].tolist()}

    if "activation_values" in bundle:
        values=np.asarray(bundle["activation_values"]);limit=bundle["calibration_absmax"]
        fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        axes[0].hist(values,bins=80,alpha=.6,label="Input activations")
        axes[0].axvline(limit,color="gray",ls="--");axes[0].axvline(-limit,color="gray",ls="--")
        axes[0].set(xlabel="Activation value",ylabel="Sampled elements",yscale="log",title="Sampled distribution / frozen range")
        metrics["activation"]={}
        for bits in (8,4):
            # Actual reconstruction must be supplied for measured data, never simulated silently.
            reconstruction=demo_symmetric_qdq(values,bits,limit) if demo else bundle["activation_reconstructions"][str(bits)]
            stats=activation_summary(values,limit,reconstruction);metrics["activation"][str(bits)]=stats
            diff=np.sort(np.abs(np.asarray(reconstruction)-values))
            axes[1].plot(np.log10(diff+1e-4),np.arange(1,len(diff)+1)/len(diff),label="A"+str(bits))
        axes[1].set(xlabel="log10(abs reconstruction error + 1e-4)",ylabel="Sampled-element CDF",title="A8 / A4, body and clipping tail retained")
        for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
        save(fig,"13-activation-range-and-tail")
    return files,metrics


def player_html(bundle,files):
    cs=bundle["conditions"];reference=next(c for c in cs if c["id"]=="BF16")
    # Local-only player with a single slider: space and error cursor share one step.
    data={c["id"]:{"position":c["positions_m"],"error_mm":(np.linalg.norm(np.asarray(c["positions_m"])[:min(len(c["positions_m"]),len(reference["positions_m"]))]-np.asarray(reference["positions_m"])[:min(len(c["positions_m"]),len(reference["positions_m"]))],axis=1)*1000).tolist()} for c in cs}
    payload=json.dumps(data,allow_nan=False).replace('<','\\u003c')
    options=''.join('<option>'+html.escape(c["id"])+'</option>' for c in cs if c["id"]!="BF16")
    figures=''.join('<figure><img loading="lazy" src="'+f+'" alt="'+f+'"><figcaption>'+f+'</figcaption></figure>' for f in files)
    label='合成示意，非实测 VLA 结果' if bundle['data_kind']=='synthetic_demo' else '实测诊断，按输入清单审计'
    return '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>权重与激活量化轨迹诊断</title>
<style>body{font:16px system-ui;max-width:1200px;margin:auto;padding:16px;color:#18242d;background:#fff}header{background:#fff1c9;padding:14px;position:sticky;top:0;z-index:2}img{width:100%;height:auto}figure{margin:28px 0}label,select,input{font:inherit}select{max-width:100%}input[type=range]{width:100%}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}svg{width:100%;height:auto}text{font:14px system-ui}path{fill:none;stroke-width:2}@media(max-width:680px){.pair{grid-template-columns:1fr}}</style>
<header>'''+label+'''。图片只展示诊断形式，不据此报告四 suite 成功率。</header>
<h1>轨迹与误差同步查看</h1><label>量化条件 <select id="condition">'''+options+'''</select></label>
<p id="readout" aria-live="polite"></p><input id="step" type="range" min="0" max="1" value="0" aria-label="控制步">
<div class="pair"><svg id="space" viewBox="0 0 550 330" role="img" aria-label="同一控制步的XY轨迹位置"></svg><svg id="error" viewBox="0 0 550 330" role="img" aria-label="位置偏离曲线及当前步"></svg></div>
<p>深灰为 BF16，蓝色为所选量化条件。空间位置和误差游标使用同一控制步。独立闭环路径差异包含状态分叉，不能全部归因于局部量化误差。</p>
'''+figures+'''<script>const data='''+payload+''';const sel=document.getElementById('condition'),slider=document.getElementById('step');
function draw(){const c=data[sel.value],r=data.BF16,n=Math.min(c.position.length,r.position.length,c.error_mm.length);slider.max=n-1;const k=Math.min(+slider.value,n-1);slider.value=k;
const all=c.position.concat(r.position),xmin=Math.min(...all.map(p=>p[0])),xmax=Math.max(...all.map(p=>p[0])),ymin=Math.min(...all.map(p=>p[1])),ymax=Math.max(...all.map(p=>p[1]));
const x=v=>55+(v-xmin)/Math.max(xmax-xmin,1e-9)*465,y=v=>270-(v-ymin)/Math.max(ymax-ymin,1e-9)*225;
const path=a=>a.map((p,i)=>(i?'L':'M')+x(p[0])+','+y(p[1])).join(' ');
document.getElementById('space').innerHTML='<path d="M55,40 V270 H520" stroke="#8a9399"/><path d="'+path(r.position)+'" stroke="#59636a"/><path d="'+path(c.position)+'" stroke="#2178bc"/><circle cx="'+x(r.position[k][0])+'" cy="'+y(r.position[k][1])+'" r="5" fill="#59636a"/><circle cx="'+x(c.position[k][0])+'" cy="'+y(c.position[k][1])+'" r="5" fill="#2178bc"/><text x="55" y="25">XY 末端路径</text><text x="55" y="305">x: '+(xmin*1000).toFixed(0)+' 至 '+(xmax*1000).toFixed(0)+' mm</text><text x="55" y="325">y: '+(ymin*1000).toFixed(0)+' 至 '+(ymax*1000).toFixed(0)+' mm</text>';
const em=Math.max(...c.error_mm,1e-9),xx=i=>55+i/Math.max(n-1,1)*465,yy=v=>270-v/em*225,ep=c.error_mm.map((v,i)=>(i?'L':'M')+xx(i)+','+yy(v)).join(' ');
document.getElementById('error').innerHTML='<path d="M55,40 V270 H520" stroke="#8a9399"/><path d="'+ep+'" stroke="#2178bc"/><path d="M'+xx(k)+',40 V270" stroke="#e08b36"/><circle cx="'+xx(k)+'" cy="'+yy(c.error_mm[k])+'" r="5" fill="#2178bc"/><text x="55" y="25">位置偏离 0 至 '+em.toFixed(1)+' mm</text><text x="55" y="305">控制步 0 至 '+(n-1)+'</text>';
document.getElementById('readout').textContent='控制步 '+k+'，末端位置差 '+c.error_mm[k].toFixed(2)+' mm';}
sel.addEventListener('change',draw);slider.addEventListener('input',draw);draw();</script>'''


def main():
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo",action="store_true");mode.add_argument("--bundle",type=Path)
    p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError("Fresh output directory required")
    bundle=synthetic_bundle() if args.demo else json.loads(args.bundle.read_text(encoding="utf-8"))
    if not args.demo:
        if bundle.get("data_kind")!="measured":raise ValueError("Synthetic bundle requires --demo")
        admission=args.bundle.parent/bundle["analysis_admission_file"]
        if hashlib.sha256(admission.read_bytes()).hexdigest()!=bundle["analysis_admission_sha256"]:raise ValueError("Analysis admission changed")
        gate=json.loads(admission.read_text(encoding="utf-8"))
        if gate.get("gate")!="PASS_WA_SAME_INPUT_ANALYSIS_V2" or not all(gate.get(k) is True for k in ("same_input","same_coordinate","frozen_scales","coverage_audited")):
            raise ValueError("Measured W/A analysis has not passed lineage/coordinate checks")
        content={k:v for k,v in bundle.items() if k not in ("analysis_admission_file","analysis_admission_sha256")}
        content_sha=hashlib.sha256(json.dumps(content,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
        if gate.get("bundle_content_sha256")!=content_sha:raise ValueError("Measured bundle changed after analysis admission")
    validate_bundle(bundle);args.output.mkdir(parents=True)
    files,metrics=plot_bundle(bundle,args.output)
    (args.output/"metrics.json").write_text(json.dumps(metrics,allow_nan=False,indent=2)+'\n',encoding="utf-8")
    (args.output/"index.html").write_text(player_html(bundle,files),encoding="utf-8")
    print(json.dumps(dict(data_kind=bundle["data_kind"],figures=len(files),measured_episodes=0 if args.demo else "see audited episode manifest")))


if __name__=="__main__":main()
