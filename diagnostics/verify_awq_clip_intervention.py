"""CPU audit: derived profile may change only the declared clip tensors."""
import argparse,hashlib,json
from pathlib import Path
import torch

def equal(a,b):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    elif isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert type(a)==type(b) and len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
p.add_argument('--derived',type=Path,required=True);a=p.parse_args()
s=torch.load(a.source,map_location='cpu',weights_only=True)
d=torch.load(a.derived,map_location='cpu',weights_only=True)
i=d['diagnostic_intervention'];removed=set(i['removed_language_clips'])
assert hashlib.sha256(a.source.read_bytes()).hexdigest()==i['source_profile_sha256']
assert len(removed)=={'all':160,'attention':64,'mlp':96}[i.get('removed_clip_family','all')]
assert s.keys()==(d.keys()-{'diagnostic_intervention'})
for key in s:
    if key!='entries':equal(s[key],d[key])
assert s['entries'].keys()==d['entries'].keys()
for name,entry in s['entries'].items():
    other=d['entries'][name];assert entry.keys()==other.keys()
    for key,value in entry.items():
        if key=='clip_max' and name in removed:
            assert isinstance(value,torch.Tensor) and other[key] is None
        else:equal(value,other[key])
result=dict(source_profile_sha256=i['source_profile_sha256'],
    derived_profile_sha256=hashlib.sha256(a.derived.read_bytes()).hexdigest(),
    removed_clip_family=i.get('removed_clip_family','all'),removed_clips=len(removed),
    all_other_fields_exactly_equal=True,targets=len(s['entries']),block_scales=len(s['block_scales']))
a.derived.with_suffix('.verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
