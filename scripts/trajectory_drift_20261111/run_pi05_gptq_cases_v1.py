"""Execute exact preregistered independent-Torch cases; reject repeats and wrong roles."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from qvla.infra.common import digest,write_json,now
from qvla.infra.lifecycle import quota_decision

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--round-dir',type=Path,required=True)
p.add_argument('--suite')
p.add_argument('--activation-bits',type=int,choices=[16],required=True)
p.add_argument('--role',choices=['development','calibration'],required=True)
a=p.parse_args()
s=json.loads((a.round_dir/'series.json').read_text())
admission=Path(s['bf16_admission'])
profile=Path(s['profile'])
profile_result=json.loads(profile.with_name('result.json').read_text())
assert profile_result['precision_gates'].get(str(a.activation_bits))=='PASS_RELOAD_FINITE'
assert json.loads(admission.read_text())['gate']=='PASS_PI05_TORCH_INDEPENDENT_BF16_RELOAD_INPUT'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='4')
for row in s['cases']:
    if row['role']!=a.role or (a.suite and row['suite']!=a.suite):continue
    if quota_decision(json.loads((a.round_dir/'quota.json').read_text()),s['control'])!='CONTINUE':
        raise RuntimeError('No quota for another case')
    assert shutil.disk_usage(a.root).free>=8*1024**3
    pp=a.round_dir/'case-plans'/(row['round_id']+'.json')
    # Bind the newly produced fixed admission as a runtime input without changing the frozen case template.
    template=json.loads(pp.read_text());assert digest(pp)==row['template_sha256']
    template.update(admission_path=str(admission),admission_sha256=digest(admission),profile_sha256=digest(profile))
    actual=pp.with_name(pp.stem+'-runtime.json');assert not actual.exists();write_json(actual,template)
    args=[sys.executable,'-u','-m','qvla.execute_pi05_torch_gptq_episode_v1','--checkpoint',str(a.root/'models/pi05_libero_pytorch_bridge_v2'),
          '--openpi-source',str(a.root/'sources/openpi'),'--libero-source',str(a.root/'sources/LIBERO'),
          '--libero-config',str(a.root/'libero-config-v1'),'--plan',str(actual),'--output',template['output'],
          '--round-id',row['round_id'],'--suite',row['suite'],'--task-id',str(row['task_id']),
          '--reset-index',str(row['reset_index']),'--model-seed','7','--profile',str(profile),'--activation-bits',str(a.activation_bits)]
    with (a.round_dir/'case-logs'/(row['round_id']+'.log')).open('xb') as log:
        rc=subprocess.call(args,env=env,stdout=log,stderr=subprocess.STDOUT)
    write_json(a.round_dir/'last-case-status.json',dict(time_utc=now(),case=row,exit_code=rc))
    if rc:raise RuntimeError('Independent Torch case failed: '+row['round_id'])
    from qvla.audit_pi05_torch_gptq_trace_v1 import audit
    write_json(a.round_dir/'audits'/(row['round_id']+'.json'),audit(Path(template['output']),actual))
