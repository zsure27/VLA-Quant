"""Local vision norm->linear smoothing equivalence; no model weight changes.

FP32 control uses the same BF16-stored source weights promoted to FP32.
Local first-eight-token probes do not establish end-to-end SQ baseline validity.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import torch
from probe import initialize_readonly, predict, metric, digest
from qvla.official_quant_adapter import load_official_functions, smooth_groups

@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--profile", required=True, type=Path)
    p.add_argument("--sample", required=True, type=Path)
    p.add_argument("--official-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a=p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cfg,model,head,proprio,processor=initialize_readonly(a.checkpoint,7)
    from qvla.run_eval_official_quant import load_profiles
    from qvla.baseline_contract import checkpoint_identity
    entries,profile=load_profiles([a.profile], "smoothquant")
    if profile["checkpoint_identity"] != checkpoint_identity(a.checkpoint):
        raise ValueError("profile checkpoint content mismatch")
    alpha=profile.get("smooth_alpha")
    if alpha is None: raise ValueError("profile smoothing alpha missing")
    functions=load_official_functions(a.official_root)
    modules=dict(model.named_modules())
    groups=smooth_groups(set(entries))
    selected=[]
    for prefix in ("vision_backbone.featurizer.", "vision_backbone.fused_featurizer."):
        branch=[g for g in groups if g[0].startswith(prefix)]
        indices=sorted({int(g[0].split(".blocks.")[1].split(".")[0]) for g in branch})
        stages={indices[0],indices[len(indices)//2],indices[-1]}
        selected.extend(g for g in branch if int(g[0].split(".blocks.")[1].split(".")[0]) in stages)
    captured={}
    handles=[]
    def hook(name):
        def capture(_module,args):
            if name not in captured:
                x=args[0].detach().reshape(-1,args[0].shape[-1])[:8].cpu().clone()
                captured[name]=x
        return capture
    for norm,_,_ in selected: handles.append(modules[norm].register_forward_pre_hook(hook(norm)))
    try: predict(a.sample,cfg,model,head,proprio,processor)
    finally:
        for h in handles: h.remove()
    if set(captured)!={g[0] for g in selected}: raise RuntimeError("selected norms not captured")
    rows=[]
    for norm_name,target_names,scale_name in selected:
        if len(target_names)!=1: raise ValueError("vision local probe expects one target per group")
        target_name=target_names[0]
        act=entries[scale_name]["activation_absmax"].float()
        x=captured[norm_name].cuda()
        for case,dtype,smooth_dtype in (("fp32",torch.float32,torch.float32),
                                       ("bf16",torch.bfloat16,torch.bfloat16),
                                       ("fp32_transform_cast_bf16",torch.bfloat16,torch.float32)):
            norm=copy.deepcopy(modules[norm_name]).to(dtype=dtype)
            fc=copy.deepcopy(modules[target_name]).to(dtype=dtype)
            reference=fc(norm(x.to(dtype)))
            norm.to(dtype=smooth_dtype); fc.to(dtype=smooth_dtype)
            functions["smooth_ln_fcs"](norm,[fc],act,alpha)
            norm.to(dtype=dtype); fc.to(dtype=dtype)
            candidate=fc(norm(x.to(dtype)))
            result=metric(reference.float().cpu(),candidate.float().cpu())
            rows.append({"norm":norm_name,"linear":target_name,"case":case,"tokens":x.shape[0],
                         "input_space":"first_camera_teacher_first_8_flattened_tokens",**result})
            del norm,fc,reference,candidate
        print("LOCAL_GROUP_COMPLETE",norm_name,flush=True)
    (a.output/"metrics.json").write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n")
    (a.output/"manifest.json").write_text(json.dumps({"sample":a.sample.name,"sample_sha256":digest(a.sample),
        "profile_sha256":digest(a.profile),"source_sha256":digest(__file__),"alpha":alpha,
        "groups":len(selected),"cases":3,"tf32":False,"torch":torch.__version__,
        "note":"Local smoothing equivalence only. No quantization, no PEFT, no end-to-end gate."},indent=2)+"\n")
    (a.output/"complete.json").write_text(json.dumps({"status":"MEASUREMENT_COMPLETE","records":len(rows)})+"\n")

if __name__=="__main__": main()
