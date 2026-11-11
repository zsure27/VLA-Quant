"""Admit an independent same-PyTorch BF16 reference, not a JAX conversion gate."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from qvla.infra.common import digest,write_json,now
from qvla.pi05_torch_backend_v1 import BACKEND_ID,SOURCE_COMMIT,load_policy,target_names,noise_stream,next_noise

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
report=dict(gate='IN_PROGRESS_PI05_TORCH_INDEPENDENT_BF16',backend=BACKEND_ID,
            jax_equivalence_gate_passed=False,started_at=now())
t0=time.monotonic()
try:
    assert subprocess.check_output(['git','-C',str(a.root/'sources/openpi'),'rev-parse','HEAD'],text=True).strip()==SOURCE_COMMIT
    weight=a.root/'models/pi05_libero_pytorch_bridge_v2/model.safetensors'
    expected=(a.root/'checks/pi05-pytorch-bridge-v2.sha256').read_text().split()[0]
    assert digest(weight)==expected
    query=a.root/'runs/20261108-216-pi05-spatial-bf16-task0-reset5-v2/query-00000.npz'
    assert digest(query)=='d54325a01dbbbd5dd5e983d472d3c522e6a98cfbb40f8e5b24eb1fd36a7fc67c'
    with np.load(query,allow_pickle=False) as q:
        obs={'observation/image':q['image'],'observation/wrist_image':q['wrist_image'],
             'observation/state':q['state'],'prompt':str(q['prompt'].item())}
    noise=next_noise(noise_stream())
    policy=load_policy(a.root)
    assert all(torch.isfinite(x).all().item() for x in policy._model.parameters())
    names=target_names(policy._model)
    shapes={n:list(dict(policy._model.named_modules())[n].weight.shape) for n in names}
    # Full model scope is audited here, not inherited from the older incomplete 333-name inventory.
    assert len(names)==414,(len(names),'inspect scope before admitting generic quantizers')
    original=np.asarray(policy.infer(obs,noise=noise)['actions'])
    repeat=np.asarray(policy.infer(obs,noise=noise)['actions'])
    assert original.shape==(50,7) and np.isfinite(original).all()
    assert np.array_equal(original,repeat),'Same-input/noise repeat differs'
    del policy;gc.collect();torch.cuda.empty_cache()
    reloaded=load_policy(a.root)
    restored=np.asarray(reloaded.infer(obs,noise=noise)['actions'])
    assert np.array_equal(original,restored),'Reloaded same-input/noise actions differ'
    assert digest(weight)==expected
    np.savez_compressed(a.output/'input-and-actions.npz',image=obs['observation/image'],
                        wrist_image=obs['observation/wrist_image'],state=obs['observation/state'],
                        prompt=np.asarray(obs['prompt']),noise=noise,actions=original)
    report.update(gate='PASS_PI05_TORCH_INDEPENDENT_BF16_RELOAD_INPUT',same_input_repeat_exact=True,
                  reload_exact=True,all_parameters_frozen=True,all_parameters_finite=True,
                  checkpoint_sha256=expected,target_count=len(names),target_shapes=shapes,
                  input_and_action_sha256=digest(a.output/'input-and-actions.npz'),
                  protocol='LIBERO per-episode seed7; per-query NumPy default_rng(0) noise; 50x32 flow block, execute first5x7; official PyTorch eager math unchanged',
                  interpretation='Admits only the independent PyTorch BF16 cohort. Historical JAX comparisons remain failed and separate.')
except BaseException as exc:
    report.update(gate='FAIL_PI05_TORCH_INDEPENDENT_BF16',error_type=type(exc).__name__,error=str(exc))
    raise
finally:
    report.update(ended_at=now(),wall_seconds=time.monotonic()-t0)
    write_json(a.output/'result.json',report)
print(json.dumps({k:v for k,v in report.items() if k!='target_shapes'}))
