"""Original OmniQuant weight-only LWC, native OFT block reconstruction.

Four fixed reset0 frames; 20 epochs, lr .01, no LET for this weight-only
recipe. Teacher outputs are capped at 2GiB in RAM; no activation disk spool.
"""
import copy, re, math
import signal
import argparse, gc, hashlib, json, os, sys, time, subprocess, shutil
from pathlib import Path
import torch
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-groups',type=int,default=83);p.add_argument('--resume-partial',type=Path);a=p.parse_args();root=a.root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);start=time.time()
    assert subprocess.check_output(['hostname'],text=True).strip()=='autodl-container-c49843b5ba-28bcee5c'
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Another GPU policy is active; do not run concurrent models'
    assert shutil.disk_usage(root).free>=12*1024**3
    code=root/'code-cpu-prep-20261109-v1';sys.path[:0]=[str(root/'method-code/20261110-omniquant-weight-v1'),str(code),str(root/'sources/openvla-oft'),str(root/'sources/LIBERO')]
    from qvla.action_jacobian_batch import initialize,load_sample,prepare_inputs
    from qvla.baseline_contract import CONTRACT_ID,canonical_targets,checkpoint_identity,sample_identity
    # qvla is a regular package; the independent adapter is imported by path.
    import importlib.util
    spec=importlib.util.spec_from_file_location('omni_weight_adapter',root/'method-code/20261110-omniquant-weight-v1/omni_oft_weight_v1.py');adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    source=root/'sources/OmniQuant-archive-v1';source_receipt=json.load(open(root/'checks/20261109-053-quant-source-archives-v1.json'));pin=next(x for x in source_receipt if x['repository']=='OpenGVLab/OmniQuant')
    for name in ['quantize/int_linear.py','quantize/quantizer.py']:assert hashlib.sha256((source/name).read_bytes()).hexdigest()==pin['source_files_sha256'][name]
    sys.path.insert(0,str(source));from quantize.int_linear import QuantLinear
    from quantize.quantizer import UniformAffineQuantizer
    report={'gate':'RUNNING_ORIGINAL_OMNI_LWC_CALIBRATION','full_model_admitted':False,'targets_completed':0,'max_groups':a.max_groups,'entries':{},'source_commit':pin['requested_commit']}
    def status():
        (out/'status.json').write_text(json.dumps({k:v for k,v in report.items() if k!='entries'},indent=2))
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Control stop signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
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
        names=set(canonical_targets());modules=dict(model.named_modules());groups=[]
        for name,module in model.named_modules():
            if name.endswith('.patch_embed.proj') or re.fullmatch(r'vision_backbone\.(?:fused_)?featurizer\.blocks\.\d+|language_model\.model\.layers\.\d+',name):
                targets=[n for n in names if n==name or n.startswith(name+'.')]
                if targets:groups.append((name,module,sorted(targets)))
        assert len(groups)==83 and sum(len(g[2]) for g in groups)==422
        chosen=groups[:a.max_groups];assert 1<=len(chosen)<=83
        def tree(value,device):
            if isinstance(value,torch.Tensor):return value.detach().to(device).clone()
            if isinstance(value,tuple):return tuple(tree(x,device) for x in value)
            if isinstance(value,list):return [tree(x,device) for x in value]
            if isinstance(value,dict):return {k:tree(v,device) for k,v in value.items()}
            assert value is None or isinstance(value,(str,bool,int,float)),type(value)
            return value
        def tensor(value):return value[0] if isinstance(value,tuple) else value
        teachers={n:[] for n,_,_ in chosen};bytes_cached=[0];handles=[]
        def teacher_hook(name):
            def hook(module,args,kwargs,output):
                y=tensor(output).detach().cpu().clone();bytes_cached[0]+=y.numel()*y.element_size();assert bytes_cached[0]<=2*1024**3,'Teacher cache exceeds explicit 2GiB cap';teachers[name].append(y)
            return hook
        for name,module,_ in chosen:handles.append(module.register_forward_hook(teacher_hook(name),with_kwargs=True))
        try:
            for item in prepared:forward(item)
        finally:
            for handle in handles:handle.remove()
        report['teacher_cache_bytes']=bytes_cached[0];report['epochs']=20;report['lwc_lr']=.01;report['LET']=False;status()
        resume_groups=0
        if a.resume_partial:
            partial=torch.load(a.resume_partial,map_location='cpu',weights_only=True);previous=json.load(open(a.resume_partial.parent/'status.json'))
            assert partial['gate']=='INCOMPLETE_EXCLUDED_NOT_BASELINE' and partial['source_commit']==pin['requested_commit']
            resume_groups=previous['groups_completed'];assert resume_groups==25 and previous['targets_completed']==94
            expected={target for _,_,targets in chosen[:resume_groups] for target in targets};assert set(partial['entries'])==expected
            for target,entry in partial['entries'].items():
                m=modules[target];m.weight.copy_(adapter.unpack(entry,m.weight.device,m.weight.dtype))
            report['entries']=partial['entries'];report['group_costs']=previous['group_costs'];report['resume_prefix_groups']=resume_groups;report['resume_prefix_targets']=94
            h=hashlib.sha256()
            with a.resume_partial.open('rb') as f:
                for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
            report['resume_partial_sha256']=h.hexdigest();status()
        for index,(name,module,targets) in enumerate(chosen):
            if index<resume_groups:teachers.pop(name);continue
            captured=[]
            def capture(_module,args,kwargs,output):captured.append((tree(args,'cpu'),tree(kwargs,'cpu'),tensor(output).detach().cpu().clone()))
            handle=module.register_forward_hook(capture,with_kwargs=True)
            try:
                for item in prepared:forward(item)
            finally:handle.remove()
            assert len(captured)==len(teachers[name])==(8 if name.startswith('vision_backbone.') else 4)
            qblock=copy.deepcopy(module).eval().requires_grad_(False);wrapped={}
            if isinstance(module,torch.nn.Conv2d):
                qblock=adapter.QuantConvLWC(module,UniformAffineQuantizer);wrapped[name]=qblock
            else:
                for target in targets:
                    relative=target[len(name)+1:];parent_path,_,leaf=relative.rpartition('.');parent=qblock.get_submodule(parent_path) if parent_path else qblock;old=getattr(parent,leaf);assert isinstance(old,torch.nn.Linear)
                    quant=QuantLinear(old,weight_quant_params=dict(n_bits=4,per_channel_axes=[0],symmetric=False,dynamic_method='per_channel',group_size=math.gcd(128,old.in_features),lwc=True),act_quant_params=dict(n_bits=16,dynamic_method='per_token'));setattr(parent,leaf,quant);wrapped[target]=quant
            qblock=qblock.to(device)
            with torch.no_grad():
                for args,kwargs,expected in captured:
                    actual=tensor(qblock(*tree(args,device),**tree(kwargs,device)));assert torch.equal(actual.cpu(),expected),('Disabled-quant block bridge failed',name)
            qblock.float()
            for quant in wrapped.values():quant.set_quant_state(True,False)
            params=[p for p in qblock.parameters() if p.requires_grad];assert params and all('bound_factor' in n for n,p in qblock.named_parameters() if p.requires_grad)
            optimizer=torch.optim.AdamW(params,lr=.01,weight_decay=0);losses=[];nonzero=False
            for epoch in range(20):
                epoch_losses=[]
                for (args,kwargs,_),teacher in zip(captured,teachers[name]):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):prediction=tensor(qblock(*tree(args,device),**tree(kwargs,device)))
                    loss=(prediction.float()-teacher.to(device).float()).square().mean();assert torch.isfinite(loss);loss.backward()
                    grads=[p.grad for p in params];assert all(g is not None and torch.isfinite(g).all() for g in grads);nonzero=nonzero or any(bool(g.abs().sum()>0) for g in grads);optimizer.step();epoch_losses.append(float(loss.detach()))
                losses.append(sum(epoch_losses)/len(epoch_losses))
            assert nonzero,('No LWC gradient',name)
            with torch.no_grad():
                for target,quant in wrapped.items():
                    value,entry=adapter.export_weight(quant);modules[target].weight.copy_(value);report['entries'][target]=entry
            report.setdefault('group_costs',[]).append(dict(group=name,examples=len(captured),loss_first=losses[0],loss_last=losses[-1],gradient_finite_nonzero=True,disabled_quant_exact=True))
            teachers.pop(name);del qblock,wrapped,optimizer,params,grads,captured;gc.collect();torch.cuda.empty_cache();report.update(groups_completed=index+1,targets_completed=len(report['entries']),current_group=name,elapsed_s=time.time()-start);status();print('OMNI_LWC_GROUP',index+1,len(chosen),name,flush=True)
        del teachers;actions=[forward(x) for x in prepared]
        payload={'format':'OFT_ORIGINAL_OMNI_LWC_PACK_V1','contract_id':CONTRACT_ID,'method':'omniquant','bits':4,'activation_bits':16,'checkpoint_identity':identity,'sample_sha256':sample_identity(sorted(samples.glob('sample-*.npz'))),'source_commit':pin['requested_commit'],'entries':report['entries'],'expected_calibration_actions':actions,'group_size_linear':'gcd(128,in_features), maximum128; explicit nondivisible SigLIP widths','group_size_conv':'per-row original asymmetric LWC nondivisible-width handling','LET':False,'epochs':20,'lwc_lr':.01,'weight_decay':0,'calibration_scope':'four frozen reset0 frames; OFT adaptation, not full paper reproduction','kernel':'BF16 dequantized-weight execution; no INT4 speed claim'}
        path=out/'profile.tmp.pt';torch.save(payload,path);loaded=torch.load(path,map_location='cpu',weights_only=True)
        for name,entry in loaded['entries'].items():m=modules[name];m.weight.copy_(adapter.unpack(entry,m.weight.device,m.weight.dtype))
        reloaded=[forward(x) for x in prepared];assert all(torch.equal(x,y) for x,y in zip(actions,reloaded));gate='PASS_FULL_422_OMNI_LWC_PACK_AND_RELOAD' if len(report['entries'])==422 else 'PASS_PARTIAL_OMNI_LWC_ONLY_NOT_BASELINE';payload['gate']=gate;torch.save(payload,path);final=out/'profile.pt';path.replace(final)
        h=hashlib.sha256()
        with final.open('rb') as stream:
            for b in iter(lambda:stream.read(4*1024**2),b''):h.update(b)
        report.update(gate=gate,profile_bytes=final.stat().st_size,profile_sha256=h.hexdigest(),action_reload_exact=True,elapsed_s=time.time()-start)
    except BaseException as e:
        report.update(gate='FAIL_ORIGINAL_OMNI_LWC_CALIBRATION',error_type=type(e).__name__,error=str(e))
        if report['entries']:torch.save({'gate':'INCOMPLETE_EXCLUDED_NOT_BASELINE','source_commit':report['source_commit'],'entries':report['entries'],'targets_completed':report['targets_completed']},out/'partial-packed-entries.pt')
        raise
    finally:status()

if __name__=='__main__':main()
