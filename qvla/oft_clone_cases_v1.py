"""Clone-safe administrative binding around unchanged, previously admitted OFT collectors."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np
from qvla.infra.common import digest,read_json,write_json,now


def first_input(folder):
    files=sorted(folder.glob('episodes/*/query-*.npz'))
    if not files:raise ValueError('Missing actual first query')
    with np.load(files[0],allow_pickle=False) as row:
        return {key:row[key].copy() for key in ['image','wrist_image','state','instruction']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--round-dir',type=Path,required=True);p.add_argument('--suite',required=True);a=p.parse_args()
    root,rd=a.root.resolve(),a.round_dir.resolve();series=read_json(rd/'series.json')
    if socket.gethostname()!=series['hostname']:raise ValueError('Clone round hostname differs')
    if subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()!=series['gpu_uuid']:
        raise ValueError('Clone GPU UUID differs')
    for path,expected in series['scientific_files'].items():
        if digest(root/path)!=expected:raise ValueError('Frozen scientific source changed: '+path)
    legacy=root/'code-cpu-prep-20261109-v1'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',MUJOCO_GL='egl',OMP_NUM_THREADS='4',
             PYTHONPATH=os.pathsep.join(map(str,[legacy,root/'sources/openvla-oft',root/'sources/LIBERO'])))
    state=dict(suite=a.suite,state='RUNNING',completed=[],science_unchanged=True)
    progress=rd/(a.suite+'-cases.json');baselines={}
    for row in series['inherited_completed']:
        if row['condition']=='bf16' and row['reset_index']==15:
            baselines[(row['suite'],row['task_id'])]=Path(row['result_path']).parent
    (rd/'case-runs').mkdir(exist_ok=True);(rd/'case-logs').mkdir(exist_ok=True)
    cases=[row for row in series['cases'] if row['suite']==a.suite]
    for index,case in enumerate(cases):
        if (rd/'STOP_NEW_GPU').exists():
            state.update(state='STOPPED_NO_NEW_CASES');write_json(progress,state);return
        from qvla.infra.lifecycle import quota_decision
        if quota_decision(read_json(rd/'quota.json'),read_json(rd/'config.json')['control'])!='CONTINUE':
            state.update(state='STOPPED_QUOTA_NO_NEW_CASES');write_json(progress,state);return
        condition=case['condition'];method=condition.split('_w')[0] if condition!='bf16' else 'bf16'
        name=f"{rd.name}-{condition}-{case['suite']}-task{case['task_id']}-reset{case['reset_index']}"
        output=rd/'case-runs'/name
        if output.exists():raise FileExistsError('Never overwrite or resume an old output')
        route=series['drivers'][condition]
        command=[str(root/'envs/oft/bin/python'),'-u']+(['-m',route['path']] if route['kind']=='module' else [str(root/route['path'])])
        command+=['--oft-source',str(root/'sources/openvla-oft'),'--checkpoint',str(root/'models/openvla_oft'),
                  '--suite',case['suite'],'--task-id',str(case['task_id']),'--initial-state-index',str(case['reset_index']),
                  '--model-seed','7','--round-id',name,'--output',str(output)]
        if method!='bf16':
            command+=['--profile',str(root/series['profiles'][condition]['path']),
                      '--official-root',str(root/'sources/official-quantization-pinned-v2'),
                      '--method',method,'--activation-bits',condition.rsplit('a',1)[1]]
        if method in ['gptq','omniquant']:command+=['--calibration-dir',str(root/'calibration/20261109-four-suite-reset0-v1')]
        state.update(current_case=case,current_index=index,argv=command,updated_at=now());write_json(progress,state)
        started=time.monotonic()
        with (rd/'case-logs'/(name+'.log')).open('xb') as log:
            process=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            state['policy_pid']=process.pid;write_json(progress,state);code=process.wait()
        if code!=0:
            state.update(state='CASE_PROCESS_FAILED',exit_code=code,updated_at=now());write_json(progress,state)
            raise RuntimeError('Frozen collector failed; preserve partial output: '+name)
        result=read_json(output/'result.json')
        if result['gate'] not in ['PASS_ACTUAL_OFT_QUANT_NATURAL_TRACE_PENDING_PAIRED_AUDIT','PASS_ACTUAL_OFT_BF16_NATURAL_TRACE']:
            raise ValueError('Collector gate rejected')
        if result['initial_state_sha256']!=case['initial_state_sha256']:
            raise ValueError('Actual cloned initial state differs')
        key=(case['suite'],case['task_id'])
        if method=='bf16':baselines[key]=output
        else:
            reference=baselines[key];left,right=first_input(reference),first_input(output)
            if any(not np.array_equal(left[k],right[k]) for k in left):
                raise ValueError('Cloned first actual input does not match inherited/new BF16')
        state['completed'].append(dict(case=case,result_path=str(output/'result.json'),result_sha256=digest(output/'result.json'),
                                       wall_seconds=time.monotonic()-started,first_input_paired=method!='bf16'))
        state.update(updated_at=now());write_json(progress,state)
    state.update(state='COMPLETE_PENDING_FULL_RAW_AUDIT',updated_at=now());write_json(progress,state)


if __name__=='__main__':main()
