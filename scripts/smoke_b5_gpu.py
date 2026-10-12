"""Ten real training-observation queries: bank/static parity and reload.

Inference smoke only: the experts are frozen and the small router trains on CPU.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qvla.b5_contract import require_gpu_admission,save,sha,source_hashes
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--materials',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    m=json.loads(a.materials.read_text());require_gpu_admission(m)
    for key in ('oft_root','libero_root'):sys.path.insert(0,m[key])
    import torch
    from qvla.b5_runtime import initialize
    from qvla.b5_expert_bank import FrozenExpertBank
    from qvla.extended_peft import attach_state
    from diagnostics.probe import predict
    if a.output.exists():raise ValueError('Never overwrite smoke')
    a.output.mkdir(parents=True);cfg,model,head,proprio,processor,bank,route,handles=initialize(m)
    rows=json.loads((Path(m['student80'])/'manifest.json').read_text())['samples'][:5]
    paths=[Path(m['student80'])/r['file'] for r in rows];states=bank.export_expert_states();expected={};processed={};before=bank.frozen_sha256()
    with torch.no_grad():
        for expert in (0,1):
            for i,path in enumerate(paths):
                route.begin_episode('smoke');token=route.begin_query('smoke',0,expert)
                _,value,h=predict(path,cfg,model,head,proprio,processor,state_space='policy_normalized_proprio')
                if value.shape!=(8,7) or not torch.isfinite(value).all():raise ValueError('Bad real bank action')
                expected[expert,i]=value;processed[expert,i]=h;route.end_query(token);route.end_episode()
        if before!=bank.frozen_sha256():raise ValueError('Bank mutated')
        bank.save(a.output/'bank.pt',provenance={'materials_sha256':sha(a.materials)});bank.close()
        for expert,state in enumerate(states):
            modules=dict(model.named_modules());names={k.split('.awq_recovery_lora.')[0] for k in state}
            original_hooks={n:set(modules[n]._forward_hooks) for n in names};attach_state(model,state)
            try:
                for i,path in enumerate(paths):
                    _,value,h=predict(path,cfg,model,head,proprio,processor,state_space='policy_normalized_proprio')
                    if h!=processed[expert,i] or not torch.equal(value,expected[expert,i]):raise ValueError('Real bank/static expert parity failed')
            finally:
                for n in names:
                    for key in set(modules[n]._forward_hooks)-original_hooks[n]:del modules[n]._forward_hooks[key]
                    delattr(modules[n],'awq_recovery_lora')
        reloaded=FrozenExpertBank.load(a.output/'bank.pt',model,route,expected_provenance={'materials_sha256':sha(a.materials)})
        if reloaded.frozen_sha256()!=before:raise ValueError('Real bank reload hash differs')
        reloaded.close()
    save(a.output/'receipt.json',{'gate':'PASS_GPU_B5_FIXED_EXPERT_PARITY_QUERY10','queries':10,'comparison_queries':10,
        'source_sha256':source_hashes(),'materials_sha256':sha(a.materials),'bank_sha256':before,'bank_file_sha256':sha(a.output/'bank.pt'),
        'limits':'same-observation training queries only; paired micro/first50 still required; no closed-loop claim'})
    for h in handles:h.remove()
if __name__=='__main__':main()
