"""Explicit post-search ablation: remove language clips, preserving AWQ coordinates.

Does not claim new calibration or change calibration provenance. Original input
profile is immutable; the separate output carries a reproducible intervention.
"""
import argparse,hashlib,json
from pathlib import Path
import torch

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--family',choices=['all','attention','mlp'],default='all');a=p.parse_args()
if a.output.exists() or a.source.resolve()==a.output.resolve():
    raise SystemExit('Refusing overwrite or in-place profile edits')
before=sha(a.source);d=torch.load(a.source,map_location='cpu',weights_only=True)
assert d['method']=='awq' and d['bits']==2 and d['activation_bits']==16
assert d['group_size']==64 and len(d['entries'])==422 and len(d['block_scales'])==32
language=[n for n in d['entries'] if n.startswith('language_model.model.layers.')]
assert len(language)==224
removed=[]
for name in language:
    if a.family=='attention' and '.self_attn.' not in name:continue
    if a.family=='mlp' and '.mlp.' not in name:continue
    entry=d['entries'][name]
    if entry.get('clip_max') is not None:removed.append(name)
    entry['clip_max']=None
assert len(removed)=={'all':160,'attention':64,'mlp':96}[a.family]
intervention=dict(kind='post_search_language_clip_removal',source_profile=str(a.source),
    source_profile_sha256=before,removed_language_clips=removed,language_targets=224,
    removed_clip_family=a.family,
    preserved='original native G64 block scales, calibration provenance, vision entries',
    note='Diagnostic derived profile; no new AWQ search, training, or original baseline claim.')
d['diagnostic_intervention']=intervention
a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(d,a.output)
assert sha(a.source)==before
receipt=dict(intervention,derived_profile=str(a.output),derived_profile_sha256=sha(a.output),
    derived_profile_bytes=a.output.stat().st_size,group_size=64,bits=2)
a.output.with_suffix('.intervention.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='removed_language_clips'}))
