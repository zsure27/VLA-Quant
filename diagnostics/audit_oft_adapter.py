"""CPU shape/provenance audit before any attempt to preserve original OFT LoRA.

This does not infer unmerged weights by subtraction and does not verify runtime
merge equivalence. Original checkpoint/adapter tensors are never changed.
"""
import argparse,hashlib,json,re,math
from pathlib import Path
from safetensors import safe_open
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise SystemExit('Refusing overwrite')
cfgfile=a.checkpoint/'lora_adapter/adapter_config.json'
adapter=a.checkpoint/'lora_adapter/adapter_model.safetensors'
cfg=json.loads(cfgfile.read_text());index=json.loads((a.checkpoint/'model.safetensors.index.json').read_text())['weight_map']
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
records=[];unmapped=[];count={};params={}
with safe_open(str(adapter),framework='pt',device='cpu') as sf:
    keys=set(sf.keys())
    for key in sorted(keys):
        m=re.fullmatch(r'(.+)\.lora_A(?:\.default)?\.weight',key)
        if not m:continue
        other=key.replace('.lora_A','.lora_B');assert other in keys
        stem=m.group(1);module=stem[len('base_model.model.'):] if stem.startswith('base_model.model.') else stem
        sa=sf.get_slice(key).get_shape();sb=sf.get_slice(other).get_shape()
        assert sa[0]==sb[1]
        if len(sa)==len(sb)==2:
            expected=[sb[0],sa[1]];kind='Linear'
        elif len(sa)==len(sb)==4 and sb[2:]==[1,1]:
            expected=[sb[0]]+sa[1:];kind='Conv2d'
        else:
            expected=None;kind='UNSUPPORTED'
        basekey=module+'.weight';valid=False;shape=None
        if basekey in index:
            with safe_open(str(a.checkpoint/index[basekey]),framework='pt',device='cpu') as base:
                shape=base.get_slice(basekey).get_shape()
            valid=expected is not None and shape==expected
        if not valid:unmapped.append(module)
        branch='language' if module.startswith('language_model.') else ('vision' if module.startswith('vision_backbone.') else 'other')
        n=math.prod(sa)+math.prod(sb);count[branch]=count.get(branch,0)+1;params[branch]=params.get(branch,0)+n
        records.append(dict(module=module,rank=sa[0],a_shape=sa,b_shape=sb,raw_checkpoint_key=basekey,
            raw_checkpoint_shape=shape,shape_mapping_valid=valid,parameters=n,module_kind=kind))
result=dict(adapter_config=cfg,config_sha256=sha(cfgfile),adapter_sha256=sha(adapter),
    module_counts=count,adapter_parameters=params,modules=records,unmapped_modules=unmapped,
    runtime_merge_verified=False,training_steps=0,
    note='Shape/source audit only. Must verify loaded merged weights vs raw checkpoint plus adapter before retention. Never use unverified Wmerged-delta as base.')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('modules','adapter_config')}))
