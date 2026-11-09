"""No-model-load GR00T preparation: source/asset identities and tiny CPU contracts.

Never starts inference or registers a GPU run. A passing receipt only covers the
listed CPU checks; full-model method compatibility remains a separate gate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

PINS = {'Isaac-GR00T':'4af2b622892f7dcb5aae5a3fb70bcb02dc217b96',
        'LIBERO':'8f1084e3132a39270c3a13ebe37270a43ece2a01',
        'QuantVLA-v1':'9e435e56aac4e743526557e5d9ba6e4fddd6ccca'}
SUITES = {'libero_spatial':'groot_spatial','libero_object':'groot_object',
          'libero_goal':'groot_goal','libero_10':'groot_long'}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report={'gate':'IN_PROGRESS_CPU_ONLY','gpu_run_admitted':False,'sources':{},'checkpoints':{},'tiny_contracts':[]}
    try:
        for name,pin in PINS.items():
            folder=root/'sources'/name
            head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=folder,text=True).strip()
            assert head==pin,(name,head)
            dirty=subprocess.check_output(['git','status','--porcelain'],cwd=folder,text=True)
            assert not dirty,(name,dirty)
            report['sources'][name]=head
        quant=root/'sources/QuantVLA-v1'
        source=(quant/'run_quantvla.sh').read_text()
        include=re.findall(r"^export GR00T_DUQUANT_INCLUDE='([^']+)'",source,re.M)[-1]
        exclude=re.findall(r"^export GR00T_DUQUANT_EXCLUDE='([^']+)'",source,re.M)[-1]
        inc,exc=re.compile(include),re.compile(exclude)
        consumers={k:[] for k in ('GR00T_DUQUANT','GR00T_ATM','GR00T_OHB')}
        for file in quant.rglob('*.py'):
            body=file.read_text(errors='replace')
            for key in consumers:
                if key in body:consumers[key].append(str(file.relative_to(quant)))
        report['component_source_occurrences']=consumers
        for suite,folder in SUITES.items():
            model=root/'models'/folder;receipt=json.loads((model/'DOWNLOAD_VERIFICATION.json').read_text())
            rows=receipt.get('files_verified',receipt.get('files'))
            if not isinstance(rows,list):rows=receipt.get('objects_verified')
            assert isinstance(rows,list),list(receipt)
            tensors={};count=0;total=0
            for row in rows:
                rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
                file=model/rel;assert file.is_file() and not file.is_symlink()
                assert file.stat().st_size==row['bytes'] and sha(file)==row['sha256'],str(file)
                count+=1;total+=row['bytes']
                if file.suffix=='.safetensors':
                    with file.open('rb') as f:
                        n=int.from_bytes(f.read(8),'little');assert 0<n<64*1024*1024
                        header=json.loads(f.read(n))
                    tensors.update({k:v for k,v in header.items() if k!='__metadata__'})
            targets=[k[:-7] for k,v in tensors.items() if k.endswith('.weight') and len(v['shape'])==2 and inc.search(k[:-7]) and not exc.search(k[:-7])]
            assert targets,(suite,'empty target mapping')
            report['checkpoints'][suite]={'files_verified':count,'bytes':total,'receipt_sha256':sha(model/'DOWNLOAD_VERIFICATION.json'),
                'tensor_count':len(tensors),'upstream_script_selected_targets':sorted(targets),
                'target_count':len(targets),'tensor_shapes_sha256':hashlib.sha256(json.dumps(tensors,sort_keys=True).encode()).hexdigest()}
            for method in ('DuQuant','QuantVLA'):
                for bits in (8,4):
                    config={'model':'groot_n15','suite':suite,'tasks':list(range(10)),'initial_states':[5,15,25,35,45],
                        'method':method,'weight_bits':4,'activation_bits':bits,'executable':False,
                        'source_commit':PINS['QuantVLA-v1'],'implementation_variant':'QuantVLA upstream GR00T DuQuant adaptation; original-paper equivalence not inferred',
                        'checkpoint_receipt_sha256':sha(model/'DOWNLOAD_VERIFICATION.json'),'target_names':targets,
                        'include_regex':include,'exclude_regex':exclude,'block_size':64,'lambda_smooth':.15,
                        'atm':method=='QuantVLA','ohb':method=='QuantVLA','fresh_matching_calibration_required':True,
                        'native_execute_horizon':1,'full_model_bf16_and_quant_gpu_admission':'PENDING',
                        'do_not_use_author_absolute_paths_or_share_cache_across_models':True}
                    key=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
                    config['content_identity']=key
                    (out/f'{suite}-{method.lower()}-w4a{bits}.json').write_text(json.dumps(config,indent=2))
            print('CHECKPOINT_SOURCE_SHA_PASS',suite,count,total,flush=True)
        os.environ['CUDA_VISIBLE_DEVICES']=''
        sys.path.insert(0,str(quant))
        import torch
        from gr00t.quantization.duquant_layers import DuQuantConfig,DuQuantLinear
        torch.manual_seed(7);base=torch.nn.Linear(16,8).requires_grad_(False);x=torch.randn(2,3,16)
        for wb,ab in ((0,0),(4,8),(4,4)):
            name=f'cpu_w{wb}a{ab}';folder=out/name
            cfg=DuQuantConfig(weight_bits=wb,act_bits=ab,block_size=4,block_out_size=4,
                lambda_smooth=.15,enable_permute=True,calib_batches=2,pack_dir=str(folder),row_rot_mode='restore')
            q=DuQuantLinear(base,name,cfg).requires_grad_(False);y=q(x)
            assert y.shape==(2,3,8) and torch.isfinite(y).all() and not y.is_cuda
            err=float((y-base(x)).abs().max())
            if wb==0:assert err<2e-5,err
            saved=out/(name+'.pt');torch.save(q.state_dict(),saved)
            again=DuQuantLinear(base,name,cfg).requires_grad_(False);again(x*2)
            again.load_state_dict(torch.load(saved,weights_only=True),strict=True)
            assert torch.equal(y,again(x)),'artifact reload changed tiny CPU output'
            report['tiny_contracts'].append({'wbits':wb,'abits':ab,'finite':True,'max_abs_to_unquantized':err,'reload_exact':True})
        report['gate']='PASS_GR00T_CPU_SOURCE_ASSET_AND_TINY_MODULE_CHECKS_ONLY'
    except BaseException as e:
        report.update(gate='FAIL_GR00T_CPU_PREPARATION',error_type=type(e).__name__,error=str(e))
        raise
    finally:
        (out/'CPU_RECEIPT.json').write_text(json.dumps(report,indent=2))
    print(report['gate'],flush=True)
if __name__=='__main__':main()
