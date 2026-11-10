"""Execute preregistered admitted GPTQ and OmniQuant cases serially after full artifact admission."""
import argparse, datetime, hashlib, json, os, shutil, subprocess
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--control',type=Path,required=True);a=p.parse_args();r=a.root.resolve();c=a.control.resolve();plan=json.loads((c/'plan.json').read_text())
    assert subprocess.check_output(['hostname'],text=True).strip()==plan['hostname']
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==plan['gpu_uuid']
    for rel,sha in plan['frozen_files'].items():assert digest(r/rel)==sha,rel
    code=r/'code-cpu-prep-20261109-v1'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',MUJOCO_GL='egl',OMP_NUM_THREADS='4',PYTHONPATH=os.pathsep.join(map(str,[code,r/'sources/openvla-oft',r/'sources/LIBERO'])))
    state={'round_id':c.name,'status':'RUNNING','plan_sha256':digest(c/'plan.json'),'completed':[]}
    def save():
        state['observed_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();t=c/'status.tmp';t.write_text(json.dumps(state,indent=2));t.replace(c/'status.json')
    try:
        for i,case in enumerate(plan['cases']):
            if (c/'STOP_NEW_GPU').exists():state['status']='STOPPED_BY_CONTROL';break
            quota=json.loads((r/'control/20261110-053-gptq-reset5-v1/quota-latest.json').read_text());now=datetime.datetime.now(datetime.timezone.utc);age=(now-datetime.datetime.fromisoformat(quota['observed_at'])).total_seconds()
            override=json.loads((c/'quota-first-window-override-v1.json').read_text()) if (c/'quota-first-window-override-v1.json').exists() else {}
            waived=(quota.get('primary_resets_at')==override.get('primary_resets_at_at_authorization') and override.get('kind')=='user_single_primary_window_override')
            if (quota['primary_remaining']<=15 and not waived) or age>1800:state['status']='STOPPED_QUOTA_OR_STALE_OBSERVATION';break
            assert shutil.disk_usage(r).free>=6*1024**3
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            name=f"{c.name}-{case['condition']}-{case['suite']}-task{case['task_id']}-reset{case['reset_index']}";output=r/'runs'/name;assert not output.exists()
            method=case['condition'].split('_w')[0] if case['condition']!='bf16' else 'bf16';route=plan['drivers'][case['condition']]
            cmd=[str(r/'envs/oft/bin/python'),'-u']+(['-m',route['path']] if route['kind']=='module' else [str(r/route['path'])])
            cmd+=['--oft-source',str(r/'sources/openvla-oft'),'--checkpoint',str(r/'models/openvla_oft'),'--suite',case['suite'],'--task-id',str(case['task_id']),'--initial-state-index',str(case['reset_index']),'--model-seed','7','--round-id',name,'--output',str(output)]
            if method!='bf16':cmd+=['--profile',str(r/plan['profiles'][case['condition']]['path']),'--official-root',str(r/'sources/official-quantization-pinned-v2'),'--method',method,'--activation-bits',case['condition'].rsplit('a',1)[1]]
            if method in ['gptq','omniquant']:cmd+=['--calibration-dir',str(r/'calibration/20261109-four-suite-reset0-v1')]
            state.update(current_case=case,current_index=i,argv=cmd);save()
            with (r/'logs'/(name+'.log')).open('x') as log:
                proc=subprocess.Popen(cmd,cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT);state['pid']=proc.pid;save();exitcode=proc.wait()
            assert exitcode==0,('trajectory admission or process failed',exitcode)
            result=json.loads((output/'result.json').read_text());assert result['gate'] in ['PASS_ACTUAL_OFT_QUANT_NATURAL_TRACE_PENDING_PAIRED_AUDIT','PASS_ACTUAL_OFT_BF16_NATURAL_TRACE']
            meta=json.loads((output/'episodes'/('task-%02d-reset-%03d'%(case['task_id'],case['reset_index']))/'episode.json').read_text());assert meta['init_state_sha256']==case['initial_state_sha256']
            state['completed'].append({'case':case,'result':str(output/'result.json'),'sha256':digest(output/'result.json')});save()
        else:state['status']='COMPLETE_PENDING_PAIRED_AUDIT'
    except BaseException as e:state.update(status='FAILED_STOPPED',error_type=type(e).__name__,error=str(e));raise
    finally:save()

if __name__=='__main__':main()
