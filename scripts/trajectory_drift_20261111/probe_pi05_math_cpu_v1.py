"""Small BF16 attention precision probe; not a full-model admission test."""
import argparse
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import torch
import jax
import jax.numpy as jnp

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
assert jax.default_backend()=='cpu' and not torch.cuda.is_available()
from transformers.models.gemma import modeling_gemma
rng=np.random.default_rng(7)
q=rng.normal(size=(1,2,5,16)).astype('float32')
k=rng.normal(size=(1,2,7,16)).astype('float32')
v=rng.normal(size=(1,2,7,16)).astype('float32')
tq,tk,tv=[torch.tensor(x).to(torch.bfloat16) for x in (q,k,v)]
jq,jk=[jnp.asarray(x,dtype=jnp.bfloat16) for x in (q,k)]
scale=16**-0.5
# JAX source explicitly scales BF16 q before an FP32-result qk contraction.
jlog=np.asarray(jnp.einsum('bhtd,bhsd->bhts',jq*scale,jk,preferred_element_type=jnp.float32))
taligned=torch.matmul((tq*scale).float(),tk.float().transpose(-1,-2)).numpy()
teager=(torch.matmul(tq,tk.transpose(-1,-2))*scale).float().numpy()
report=dict(gate='PASS_PI05_SMALL_OPERATOR_PROBE_CPU',seed=7,shape=list(q.shape),
    source_eager_attention=inspect.getsource(modeling_gemma.eager_attention_forward),
    fp32_result_qk_vs_jax_max_abs=float(np.max(np.abs(taligned-jlog))),
    bf16_result_qk_vs_jax_max_abs=float(np.max(np.abs(teager-jlog))),
    full_model_loaded=False,gpu_used=False,full_model_bridge_admitted=False,
    interpretation='Identifies an operator-level precision hypothesis only; does not attribute the observed action bridge failure or authorize quantized rollouts.')
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='source_eager_attention'}))
