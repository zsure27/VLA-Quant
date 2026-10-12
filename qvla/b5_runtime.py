"""Shared GPU initialization and offline same-observation expert queries."""
from __future__ import annotations
import inspect
from pathlib import Path
import numpy as np
import torch
from qvla.b5_contract import sha,candidate_contract
from qvla.b5_expert_bank import FrozenExpertBank
from qvla.b5_route_state import ChunkRouteState
def initialize(materials,teacher=False):
    if not torch.cuda.is_available():raise RuntimeError('Full-model B5 execution requires a user-enabled GPU')
    import shutil
    if shutil.disk_usage(Path(materials['checkpoint']).parent).free<6*1024**3:raise RuntimeError('Persistent archive reserve below 6GiB')
    from diagnostics.probe import initialize_readonly
    from qvla.run_eval_official_quant import load_profiles,apply_quantization
    from qvla.official_quant_adapter import load_official_functions
    from diagnostics.awq_interventions import attention_no_clip_primary_g64_plan
    from qvla.model_registry import validate_effective_plan
    records=candidate_contract(materials)
    entries,meta=load_profiles([Path(materials['w2_g128'])],'awq')
    if meta['checkpoint_identity']!=records[0]['checkpoint_identity']:raise ValueError('Quantization/expert checkpoint mismatch')
    cfg,model,head,proprio,processor=initialize_readonly(materials['checkpoint'],7)
    torch.backends.cuda.matmul.allow_tf32=False
    if model.language_model.config._attn_implementation!=meta['llm_attention']:raise ValueError('OFT attention implementation changed')
    if sha(inspect.getsourcefile(type(model)))!=meta['model_source_sha256']:raise ValueError('Real model class source differs')
    handles=[];bank=None;route=None
    if not teacher:
        peer=load_profiles([Path(materials['w2_g64'])],'awq')
        plan=attention_no_clip_primary_g64_plan(entries,meta,peer)
        validate_effective_plan('A4',plan[1])
        handles=apply_quantization(model,entries,meta,'awq',load_official_functions(Path(materials['official_root'])),2,16,awq_plan=plan)
        route=ChunkRouteState()
        bank=FrozenExpertBank(model,route,torch.load(materials['b3_adapter'],map_location='cpu',weights_only=True),
                              torch.load(materials['b4_adapter'],map_location='cpu',weights_only=True))
    return cfg,model,head,proprio,processor,bank,route,handles
@torch.no_grad()
def normalized_query(sample,cfg,model,head,proprio,processor,bank,route,expert):
    from qvla.action_jacobian_batch import prepare_inputs
    token=None
    if bank is not None:
        route.begin_episode('offline');token=route.begin_query('offline',0,int(expert))
    try:
        with np.load(sample,allow_pickle=False) as stored:
            observation={'image':stored['image'],'wrist_image':stored['wrist_image'],'state':stored['state'].copy(),'instruction':str(stored['instruction'].item())}
        inputs,state=prepare_inputs(observation,cfg,model,processor,torch.device('cuda:0'))
        import hashlib
        hashes={k:hashlib.sha256(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for k,v in inputs.items()}
        hashes['normalized_proprio']=hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest()
        _,hidden=model.predict_action(**inputs,unnorm_key=cfg.unnorm_key,do_sample=False,proprio=state,proprio_projector=proprio,action_head=head,noisy_action_projector=None,use_film=False)
        normalized=head.predict_action(hidden).reshape(-1,7).detach().float().cpu()
        if normalized.shape!=(8,7) or not torch.isfinite(normalized).all():raise ValueError('Nonfinite/wrong normalized chunk')
        if token is not None:route.end_query(token)
        return normalized,hashes
    except BaseException:
        if token is not None and route.active_token is not None:route.abort_query(token)
        raise
    finally:
        if bank is not None:route.end_episode()

class Router(torch.nn.Module):
    def __init__(self,task_only=False):
        super().__init__();self.task_only=task_only
        self.net=torch.nn.Sequential(torch.nn.Linear(10 if task_only else 402,64),torch.nn.ReLU(),torch.nn.Linear(64,2))
    def forward(self,x):return self.net(x[..., -10:] if self.task_only else x)
