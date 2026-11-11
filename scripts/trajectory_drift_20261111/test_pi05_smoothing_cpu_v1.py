"""Test affine folding against the actual pinned Gemma RMSNorm on CPU."""
import argparse
import copy
import json
from pathlib import Path
import torch
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm
from qvla.pi05_smoothquant_v1 import fold_norm

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
torch.manual_seed(7);rows=[]
for kind in ['layer_norm','gemma_static','gemma_adaptive']:
    x=torch.randn(2,3,16);scale=torch.exp(torch.linspace(-1,1,16));cond=torch.randn(2,8)
    if kind=='layer_norm':norm=torch.nn.LayerNorm(16)
    else:norm=GemmaRMSNorm(16,cond_dim=8 if kind=='gemma_adaptive' else None)
    for value in norm.parameters():value.data.uniform_(-.3,.3)
    fc=torch.nn.Linear(16,9);altered=copy.deepcopy(norm);head=copy.deepcopy(fc)
    if kind=='layer_norm':before=norm(x);old_gate=None
    else:before,old_gate=norm(x,cond if kind=='gemma_adaptive' else None)
    expected=fc(before)
    fold_norm(altered,scale);head.weight.data.mul_(scale[None,:])
    if kind=='layer_norm':after=altered(x);gate=None
    else:after,gate=altered(x,cond if kind=='gemma_adaptive' else None)
    actual=head(after);error=(actual-expected).abs().max().item()
    assert error<2e-6,(kind,error)
    gate_exact=old_gate is None or torch.equal(gate,old_gate);assert gate_exact
    rows.append(dict(kind=kind,max_abs=error,adaptive_gate_unchanged=gate_exact))
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(dict(gate='PASS_PI05_SQ_AFFINE_CPU',checks=rows,full_model_admitted=False),indent=2)+'\n')
print(json.dumps(rows))
