"""Compare JAX/Torch preprocessing on a saved query without loading a model."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
sys.path[:0] = [str(a.root/'sources/openpi/src'), str(a.root/'sources/openpi/packages/openpi-client/src')]
import jax
import jax.numpy as jnp
import torch
from openpi.training import config, checkpoints
from openpi import transforms
from openpi.models import model
from openpi.models_pytorch import preprocessing_pytorch

assert jax.default_backend() == 'cpu'
assert not torch.cuda.is_available()
query = a.root/'runs/20261108-216-pi05-spatial-bf16-task0-reset5-v2/query-00000.npz'
assert hashlib.sha256(query.read_bytes()).hexdigest() == 'd54325a01dbbbd5dd5e983d472d3c522e6a98cfbb40f8e5b24eb1fd36a7fc67c'
with np.load(query, allow_pickle=False) as q:
    raw = {'observation/image':q['image'], 'observation/wrist_image':q['wrist_image'],
           'observation/state':q['state'], 'prompt':str(q['prompt'].item())}
c = config.get_config('pi05_libero')
dc = c.data.create(c.assets_dirs, c.model)
ns = checkpoints.load_norm_stats(a.root/'models/pi05_libero/assets', dc.asset_id)
fn = transforms.compose([transforms.InjectDefaultPrompt(None), *dc.data_transforms.inputs,
                         transforms.Normalize(ns, use_quantiles=dc.use_quantile_norm),
                         *dc.model_transforms.inputs])
inputs = fn(jax.tree.map(lambda x:x.copy() if isinstance(x,np.ndarray) else x,raw))
j = model.Observation.from_dict(jax.tree.map(lambda x:jnp.asarray(x)[None,...], inputs))
t = model.Observation.from_dict(jax.tree.map(lambda x:torch.from_numpy(np.array(x))[None,...], inputs))
j = model.preprocess_observation(None,j,train=False)
t = preprocessing_pytorch.preprocess_observation_pytorch(t,train=False)
rows = []
def compare(name,x,y):
    x,y=np.asarray(x),y.detach().cpu().numpy()
    # Torch preprocessing uses channels-first image layout; compare the same pixels.
    layout_changed=False
    if x.ndim==4 and y.shape==(x.shape[0],x.shape[3],x.shape[1],x.shape[2]):
        y=y.transpose(0,2,3,1);layout_changed=True
    equal=x.shape==y.shape and np.array_equal(x,y)
    rows.append(dict(field=name,jax_shape=list(x.shape),torch_aligned_shape=list(y.shape),
                     layout_changed=layout_changed,exact_equal=bool(equal),
                     max_abs=float(np.max(np.abs(x.astype('float64')-y.astype('float64')))) if x.shape==y.shape else None))
assert list(j.images)==list(t.images)
for name in j.images:
    compare('image/'+name,j.images[name],t.images[name])
    compare('mask/'+name,j.image_masks[name],t.image_masks[name])
for name in ['state','tokenized_prompt','tokenized_prompt_mask']:
    compare(name,getattr(j,name),getattr(t,name))
report=dict(gate='PASS_PI05_INPUT_PREPROCESS_CPU' if all(x['exact_equal'] for x in rows) else 'FAIL_PI05_INPUT_PREPROCESS_CPU',
            query_sha256=hashlib.sha256(query.read_bytes()).hexdigest(),checks=rows,
            full_model_loaded=False,gpu_used=False,backend_action_equivalence_proven=False,
            note='Input equality does not establish full-model numerical equivalence.')
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
if report['gate'].startswith('FAIL_'):
    raise SystemExit(1)
