import json,argparse,hashlib
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument("profile",type=Path);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
d=torch.load(a.profile,map_location="cpu")
meta=d.get("metadata") or d;entries=d.get("entries",{})
if not entries and "targets" in d:entries=d["targets"]
result={"profile":str(a.profile),"top_level_keys":list(d),"metadata_keys":list(meta),"entry_count":len(entries),"note":"Readonly calibration metadata audit; no quantization or GPU experiment"}
for k in ("smooth_alpha","num_samples","seed","sample_names","group_size","format_version","method","activation_bits","bits","llm_attention"):
 if k in meta:result[k]=meta[k]
rows=[]
for name,e in entries.items():
 s=e.get("activation_absmax")
 if isinstance(s,torch.Tensor):
  x=s.detach().float().reshape(-1);rows.append({"module":name,"channels":x.numel(),"max":float(x.max()),"median_channel_absmax":float(x.median()),"max_to_median":float(x.max()/x.median().clamp_min(1e-12))})
result["activation_scale_rows"]=rows
h=hashlib.sha256()
with a.profile.open("rb") as f:
 for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
result["profile_sha256"]=h.hexdigest();a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k not in ("activation_scale_rows","sample_names")}))
