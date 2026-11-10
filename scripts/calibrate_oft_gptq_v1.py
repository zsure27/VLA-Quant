"""Original GPTQ, sequential native execution-order layers, streamed Hessians.

Uses every token of the four frozen reset0 frames, no teacher labels. Each layer
sees upstream already-quantized weights. No all-layer activation/Hessian spool.
"""
import argparse, gc, hashlib, json, os, sys, time, subprocess, shutil
from pathlib import Path
import torch
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-targets',type=int,default=422);a=p.parse_args();root=a.root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);start=time.time()
    assert subprocess.check_output(['hostname'],text=True).strip()=='autodl-container-c49843b5ba-28bcee5c'
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Another GPU policy is active; do not run concurrent models'
    assert shutil.disk_usage(root).free>=12*1024**3
    code=root/'code-cpu-prep-20261109-v1';sys.path[:0]=[str(root/'method-code/20261110-gptq-v2'),str(code),str(root/'sources/openvla-oft'),str(root/'sources/LIBERO')]
    from qvla.action_jacobian_batch import initialize,load_sample,prepare_inputs
    from qvla.baseline_contract import CONTRACT_ID,canonical_targets,checkpoint_identity,sample_identity
    # qvla is a regular package; the independent adapter is imported by path.
    import importlib.util
    spec=importlib.util.spec_from_file_location('gptq_pack_adapter',root/'method-code/20261110-gptq-v2/gptq_oft_v1.py');adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    source=root/'sources/gptq-archive-v1';source_receipt=json.load(open(root/'checks/20261109-053-quant-source-archives-v1.json'));pin=next(x for x in source_receipt if x['repository']=='IST-DASLab/gptq')
    for name in ['gptq.py','quant.py']:assert hashlib.sha256((source/name).read_bytes()).hexdigest()==pin['source_files_sha256'][name]
    sys.path.insert(0,str(source));from gptq import GPTQ
    from quant import Quantizer
    class RecordedQuantizer(Quantizer):
        def __init__(self):super().__init__();self.groups=[]
        def find_params(self,x,weight=False):
            super().find_params(x,weight=weight)
            self.groups.append((self.scale.detach().clone(),self.zero.detach().clone()))
    report={'gate':'RUNNING_ORIGINAL_GPTQ_CALIBRATION','full_model_admitted':False,'targets_completed':0,'max_targets':a.max_targets,'entries':{},'source_commit':pin['requested_commit']}
    def status():
        (out/'status.json').write_text(json.dumps({k:v for k,v in report.items() if k!='entries'},indent=2))
    try:
        status();checkpoint=root/'models/openvla_oft';identity=checkpoint_identity(checkpoint)
        from experiments.robot import openvla_utils as native
        native.update_auto_map=lambda _:None;native.check_model_logic_mismatch=lambda _:None
        cfg,model,head,proprio,processor=initialize(str(checkpoint),7);model.language_model.config.use_cache=False
        assert all(not x.requires_grad for x in model.parameters())
        # Same native input/action before any weight mutation; do not use the
        # calibration's float32-storage convention to reinterpret a raw query.
        from experiments.robot.robot_utils import get_action
        query=root/'runs/20261108-053-oft-bf16-admission-v3/episodes/task-00-reset-005/query-00000.npz'
        with np.load(query,allow_pickle=False) as z:
            observation={'full_image':z['image'].copy(),'wrist_image':z['wrist_image'].copy(),'state':z['state'].copy()}
            instruction=str(z['instruction'].item());expected=z['raw_policy_chunk'].copy()
        cfg.unnorm_key='libero_spatial_no_noops'
        actual=np.asarray(get_action(cfg,model,observation,instruction,processor=processor,action_head=head,proprio_projector=proprio,noisy_action_projector=None))
        assert np.array_equal(actual,expected),'Unquantized native first-query action bridge failed'
        report['unquantized_native_first_query_exact']=True;status()
        samples=root/'calibration/20261109-four-suite-reset0-v1';manifest=json.load(open(samples/'manifest.json'));assert manifest['gate']=='PASS_DISTINCT_RESET0_CALIBRATION_CAPTURE'
        prepared=[];device=next(model.parameters()).device
        for row in manifest['samples']:
            assert row['role']=='quant_calibration' and row['reset_index']==0
            file=samples/row['path'];assert hashlib.sha256(file.read_bytes()).hexdigest()==row['sha256']
            cfg.unnorm_key=row['suite']+'_no_noops';inp,state=prepare_inputs(load_sample(file),cfg,model,processor,device);prepared.append((inp,state,cfg.unnorm_key))
        @torch.no_grad()
        def forward(item):
            inp,state,key=item
            value=model.predict_action(**inp,unnorm_key=key,do_sample=False,proprio=state,proprio_projector=proprio,action_head=head,noisy_action_projector=None,use_film=False)
            actions=value[0] if isinstance(value,tuple) else value
            assert torch.as_tensor(actions).shape==(8,7) and torch.isfinite(torch.as_tensor(actions)).all()
            return torch.as_tensor(actions).cpu()
        reference=[forward(x) for x in prepared]
        names=set(canonical_targets());ordered=[(n,m) for n,m in model.named_modules() if n in names];assert len(ordered)==422
        selected=ordered[:a.max_targets];assert 1<=len(selected)<=422
        for index,(name,module) in enumerate(selected):
            g=GPTQ(module);q=RecordedQuantizer();q.configure(4,perchannel=True,sym=False,mse=False);g.quantizer=q;calls=[0]
            def hook(_module,args,output):g.add_batch(args[0].detach(),output.detach());calls[0]+=1
            h=module.register_forward_hook(hook)
            try:
                for item in prepared:forward(item)
            finally:h.remove()
            assert calls[0]==(8 if name.startswith('vision_backbone.') else 4),(name,calls[0])
            g.fasterquant(blocksize=128,percdamp=.01,groupsize=128,actorder=False)
            assert torch.isfinite(module.weight).all()
            # Original GPTQ first estimates full-matrix params, then each group.
            groups=q.groups[1:];count=(module.weight.flatten(1).shape[1]+127)//128;assert len(groups)==count
            entry=adapter.pack(module.weight,[x[0] for x in groups],[x[1] for x in groups],128);report['entries'][name]=entry
            del g,q,groups;gc.collect();torch.cuda.empty_cache();report.update(targets_completed=index+1,current_target=name,elapsed_s=time.time()-start);status()
            print('GPTQ_TARGET',index+1,len(selected),name,flush=True)
        actions=[forward(x) for x in prepared]
        payload={'format':'OFT_ORIGINAL_GPTQ_PACK_V1','contract_id':CONTRACT_ID,'method':'gptq','bits':4,'activation_bits':16,'checkpoint_identity':identity,'sample_sha256':sample_identity(sorted(samples.glob('sample-*.npz'))),'source_commit':pin['requested_commit'],'entries':report['entries'],'group_size':128,'percdamp':.01,'actorder':False,'expected_calibration_actions':actions,'scope':'422 native execution-order targets, original GPTQ Conv2d and Linear handling','kernel':'BF16 fake-weight execution; packed storage only'}
        path=out/'profile.tmp.pt';torch.save(payload,path);loaded=torch.load(path,map_location='cpu',weights_only=True)
        modules=dict(model.named_modules())
        for name,entry in loaded['entries'].items():
            module=modules[name];module.weight.copy_(adapter.unpack(entry,module.weight.device,module.weight.dtype))
        reloaded=[forward(x) for x in prepared];assert all(torch.equal(x,y) for x,y in zip(actions,reloaded))
        gate='PASS_FULL_422_GPTQ_PACK_AND_RELOAD' if len(selected)==422 else 'PASS_PARTIAL_GPTQ_ADMISSION_ONLY_NOT_BASELINE'
        payload['gate']=gate;torch.save(payload,path);final=out/'profile.pt';path.replace(final)
        digest=hashlib.sha256()
        with final.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024**2),b''):digest.update(block)
        report.update(gate=gate,full_model_admitted=False,profile_bytes=final.stat().st_size,profile_sha256=digest.hexdigest(),action_reload_exact=True,elapsed_s=time.time()-start)
    except BaseException as e:report.update(gate='FAIL_ORIGINAL_GPTQ_CALIBRATION',error_type=type(e).__name__,error=str(e));raise
    finally:status()

if __name__=='__main__':
    with torch.no_grad():main()
