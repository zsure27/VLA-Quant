"""Select four original calibration inputs into auditable, content-pinned manifests.

Hardlinks preserve bytes without duplicating source data; no development inputs
or labels are selected. A cache result receipt is not itself a sample manifest.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--capture',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import numpy as np
import torch
from qvla.infra.common import digest,write_json
from qvla.infra.data import audit_manifests
assert not torch.cuda.is_available()
proof=json.loads((a.capture/'result.json').read_text())
assert proof['gate']=='PASS_PI05_414_CALIBRATION_REPLAY_STATS'
assert digest(a.capture/'statistics.pt')==proof['profile_sha256']
cache=torch.load(a.capture/'statistics.pt',map_location='cpu',weights_only=True)
selected={}
for row in cache['samples']:
    assert row['role']=='calibration' and row['reset_index']==0
    selected.setdefault(row['suite'],row)
assert len(selected)==4
a.output.mkdir(parents=True,exist_ok=False)
records=[];sources=[]
for suite,row in selected.items():
    source=Path(row['path']);assert digest(source)==row['sha256'] and not source.is_symlink()
    result=source.parent/'result.json';identity=json.loads(result.read_text())
    assert identity['role']=='calibration' and identity['task_id']==0 and identity['reset_index']==0
    assert identity['condition']=='bf16' and identity['gate'].startswith('PASS_PI05_TORCH_')
    assert identity['suite']==suite and identity['backend']==cache['backend']
    folder=a.output/suite;folder.mkdir();target=folder/('query-'+row['sha256']+'.npz')
    os.link(source,target);assert digest(target)==row['sha256']
    obs=hashlib.sha256()
    with np.load(target,allow_pickle=False) as raw:
        for name in ['image','wrist_image','state','prompt']:
            value=raw[name];obs.update(name.encode());obs.update(str(value.dtype).encode());obs.update(str(value.shape).encode());obs.update(value.tobytes())
    manifest=folder/'manifest.json'
    write_json(manifest,dict(role='calibration',selection='First original cached calibration query per suite, reset0/task0 only',
                            samples=[dict(role='calibration',suite=suite,task_id=0,init_state_index=0,
                                          file=target.name,sha256=row['sha256'],observation_sha256=obs.hexdigest())]))
    records.append(dict(role='calibration',namespace='simulator_reset',suite=suite,task_ids=[0],resets=[0],manifest=str(manifest),sha256=digest(manifest)))
    sources.append(dict(suite=suite,source_query=str(source),source_query_sha256=row['sha256'],source_result=str(result),source_result_sha256=digest(result)))
cfg=dict(data=dict(calibration=records),evaluation=dict(suites=[dict(name=s,task_ids=list(range(10)),reset_offset=25,resets_per_task=1) for s in selected]))
audit=audit_manifests(cfg)
write_json(a.output/'index.json',dict(gate='PASS_SELECTED_CALIBRATION_NPZ_AND_ROLE_MANIFESTS',records=records,sources=sources,audit=audit,
                                     calibration_only=True,full_model_loaded=False,original_data_unchanged=True))
print(json.dumps(dict(gate='PASS_SELECTED_CALIBRATION_NPZ_AND_ROLE_MANIFESTS',audit=audit)))
