"""Validate the full-scope G64-attention plus W4-stage plan on real profiles."""
import argparse, json
from pathlib import Path
import torch
from diagnostics.awq_interventions import attention_visual_stage_plan, parse_layers

p=argparse.ArgumentParser(); p.add_argument('base',type=Path); p.add_argument('g64',type=Path); p.add_argument('w4',type=Path); p.add_argument('layers')
a=p.parse_args()
def load(path):
    x=torch.load(path,map_location='cpu',weights_only=True); return x['entries'],x
scales,targets,removed=attention_visual_stage_plan(*load(a.base),load(a.g64),load(a.w4),parse_layers(a.layers))
out={'targets':len(targets),'blocks':len(scales),'w4_targets':sum(v[1]==4 for v in targets.values()),
     'group64':sum(v[2]==64 for v in targets.values()),'group128':sum(v[2]==128 for v in targets.values()),
     'removed_clips':len(removed)}
assert out=={'targets':422,'blocks':32,'w4_targets':112,'group64':205,'group128':217,'removed_clips':32},out
print(json.dumps(out,sort_keys=True))
