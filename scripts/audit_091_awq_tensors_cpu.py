"""Memory-bounded audit of the actual archived AWQ profile and checkpoint.

Hashes every checkpoint byte, validates every target shape and samples four
spread weight rows per target. Does not load the 7B model or claim GPU parity.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, math, struct, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROFILE_SHA = 'ea3faca6130c88bd38fbe20be28eafc742d66605d339f8d9acfedb8da4703ea4'
QUANTIZER_SHA = '7929ee1f97addde95ce1b9c1d55f027b3b4d13fb1b9e6c850ab8a01736914e6c'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def header(path):
    with path.open('rb') as f:
        length=struct.unpack('<Q',f.read(8))[0]
        if not 0<length<20*1024*1024:raise ValueError('Invalid safetensors header length')
        return json.loads(f.read(length)),8+length

def selected_rows(root,index,name,torch,headers):
    file=root/index[name]
    if file not in headers:headers[file]=header(file)
    contents,start=headers[file]; descriptor=contents[name]
    shape=descriptor['shape']; width=math.prod(shape[1:])
    dtype={'BF16':torch.bfloat16,'F16':torch.float16,'F32':torch.float32}[descriptor['dtype']]
    element=torch.empty((),dtype=dtype).element_size()
    if descriptor['data_offsets'][1]-descriptor['data_offsets'][0]!=math.prod(shape)*element:
        raise ValueError('Safetensor offsets disagree with shape')
    rows=sorted({0,shape[0]//3,2*shape[0]//3,shape[0]-1})
    tensors=[]
    with file.open('rb') as f:
        for row in rows:
            f.seek(start+descriptor['data_offsets'][0]+row*width*element)
            raw=f.read(width*element)
            if len(raw)!=width*element:raise ValueError('Truncated weight row')
            tensors.append(torch.frombuffer(bytearray(raw),dtype=dtype).clone())
    return shape,rows,torch.stack(tensors)

def run(args):
    import torch
    torch.set_num_threads(1)
    from qvla.baseline_contract import check_scope, checkpoint_identity
    from qvla.official_quant_adapter import apply_awq_entry
    started=time.monotonic()
    if sha(args.profile)!=PROFILE_SHA:raise ValueError('Actual W4 profile SHA mismatch')
    p=torch.load(args.profile,map_location='cpu',weights_only=True)
    if (p.get('bits'),p.get('activation_bits'),p.get('group_size'),p.get('method'),p.get('algorithm'))!=(4,16,128,'awq','awq_official_llama_vision_adapter_v1'):
        raise ValueError('Unexpected quantization recipe')
    check_scope(p['entries'])
    for name,expected in p['implementation_sources'].items():
        if sha(ROOT/'qvla'/name)!=expected:raise ValueError('Archived implementation changed: '+name)
    primitive=args.quantizer.read_bytes().replace(b'\r\n',b'\n')
    if hashlib.sha256(primitive).hexdigest()!=QUANTIZER_SHA:raise ValueError('Official primitive SHA mismatch')
    tree=ast.parse(primitive.decode()); function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pseudo_quantize_tensor')
    namespace={'torch':torch}; exec(compile(ast.Module(body=[function],type_ignores=[]),str(args.quantizer),'exec'),namespace)
    quantize=namespace['pseudo_quantize_tensor']
    identity=checkpoint_identity(args.checkpoint)
    if identity!=p['checkpoint_identity']:raise ValueError('Actual checkpoint content differs from profile')
    index=json.loads((args.checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    headers={}; checked=[]
    for name,entry in sorted(p['entries'].items()):
        key=name+'.weight'
        shape,rows,original=selected_rows(args.checkpoint,index,key,torch,headers)
        if shape!=entry['shape'] or not torch.isfinite(original).all():raise ValueError('Weight/profile shape or finite mismatch: '+name)
        width=original.shape[1]; group=128; weight=original.clone()
        if name.startswith('language_model.'):
            if entry.get('recipe')!='official_llama_block_v1':raise ValueError('Language fallback recipe')
            prefix,relative=name.rsplit('.self_attn.',1) if '.self_attn.' in name else name.rsplit('.mlp.',1)
            relative=('self_attn.' if '.self_attn.' in name else 'mlp.')+relative
            for previous,followers,scale in p['block_scales'][prefix]:
                if not torch.isfinite(scale).all() or (scale<=0).any():raise ValueError('Invalid block scale')
                scale=scale.to(weight.dtype)
                if relative in followers:
                    if len(scale)!=width:raise ValueError('Column compensation mismatch')
                    weight.mul_(scale.reshape(1,-1))
                if relative==previous:
                    offset=shape[0]-len(scale)
                    for local,row in enumerate(rows):
                        if row>=offset:weight[local].div_(scale[row-offset])
            clip=entry.get('clip_max')
            if clip is not None:
                clip=clip[rows].to(weight.dtype)
                weight=weight.reshape(*clip.shape[:2],-1).clamp(-clip,clip).reshape_as(weight)
            elif not any(part in relative for part in ('q_proj','k_proj')):
                raise ValueError('Unexpected omitted language clip')
        else:
            scale=entry['input_scale'].float()
            if len(scale)!=width or not torch.isfinite(scale).all() or (scale<=0).any():raise ValueError('Invalid vision scale')
            weight=original.float()*scale.reshape(1,-1)
            padding=(-width)%group
            if padding:weight=torch.nn.functional.pad(weight,(0,padding))
            clip=entry.get('clip_max')
            if clip is not None:
                clip=clip[rows].float()
                weight=weight.reshape(*clip.shape[:2],-1).clamp(-clip,clip).reshape_as(weight)
        grouped=weight.reshape(-1,group)
        step=(grouped.amax(1,keepdim=True)-grouped.amin(1,keepdim=True)).clamp(min=1e-5)/15
        zero=(-torch.round(grouped.amin(1,keepdim=True)/step)).clamp(0,15)
        codes=(torch.round(grouped/step)+zero).clamp(0,15)
        independent=((codes-zero)*step).reshape_as(weight)
        actual=quantize(weight.clone(),n_bit=4,zero_point=True,q_group_size=128)
        if not torch.equal(actual,independent) or not torch.isfinite(actual).all():raise ValueError('Official W4 reconstruction differs: '+name)
        quantization_changed=int((actual!=weight).sum())
        if not quantization_changed:raise ValueError('No quantization-only effect: '+name)
        if name.startswith('vision_backbone.'):
            actual=(actual[:,:width]/scale.reshape(1,-1)).to(original.dtype)
            toy=torch.nn.Linear(width,len(rows),bias=False).to(original.dtype)
            with torch.no_grad():toy.weight.copy_(original)
            subset=dict(entry);subset['input_scale']=scale;subset['clip_max']=None if clip is None else clip
            apply_awq_entry(toy,subset,{'awq_quantize':quantize},4,128)
            if not torch.equal(toy.weight,actual):raise ValueError('Vision actual loader differs from reconstruction')
        changed=int((actual!=original).sum())
        if not changed:raise ValueError('No quantization effect in sampled rows: '+name)
        checked.append({'name':name,'shape':shape,'sampled_rows':rows,'code_min':int(codes.min()),'code_max':int(codes.max()),'changed_elements':changed,'quantization_only_changed_elements':quantization_changed,'primitive_equal':True})
    return {'gate':'PASS_CPU_ACTUAL_PROFILE_CHECKPOINT_AND_ROW_QUANTIZATION','profile_sha256':PROFILE_SHA,
            'checkpoint_identity':identity,'targets_checked':len(checked),'targets':checked,'official_quantizer_sha256':QUANTIZER_SHA,
            'elapsed_seconds':time.monotonic()-started,'scope':'every target shape; four spread rows per tensor; complete checkpoint file hashes',
            'limits':['No full-model forward, GPU parity, or packed INT4 kernel check.','Sampled rows do not constitute an exhaustive tensor numerical audit.','Protected parameters may change under compensated high-precision block scaling; no extra protected target quantized.']}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--quantizer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Do not overwrite a receipt')
    try:r=run(a)
    except Exception as e:r={'gate':'FAIL_CPU_ACTUAL_TENSOR_AUDIT','error_type':type(e).__name__,'error':str(e)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(r,f,indent=2,allow_nan=False)
    print(json.dumps({k:r[k] for k in ('gate','error','targets_checked','elapsed_seconds') if k in r}))
    if r['gate'].startswith('FAIL'):raise SystemExit(1)
if __name__=='__main__':main()
