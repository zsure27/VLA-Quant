"""Run only preregistered missing reset5 OFT cases, one GPU and no algorithm changes."""
import argparse, datetime, hashlib, json, os
from pathlib import Path
import shutil, subprocess, sys

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--control',type=Path,required=True);p.add_argument('--prepare',action='store_true');a=p.parse_args()
    root=a.root.resolve();c=a.control.resolve();code=root/'code-cpu-prep-20261109-v1'
    if a.prepare:
        c.mkdir(parents=True,exist_ok=False)
        assert subprocess.check_output(['hostname'],text=True).strip()=='autodl-container-c49843b5ba-28bcee5c'
        assert shutil.disk_usage(root).free>=12*1024**3
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert gpu=='GPU-f51dbeee-4a4b-a140-0a25-d9655199c7e6'
        inv=root/'checks/20261109-053-exact-resume-inventory-v1.json';data=json.loads(inv.read_text());cases=[]
        for cond in data['conditions']:
            if cond['condition'] not in ['awq_w4a16','smoothquant_w4a8','smoothquant_w4a4']:continue
            for case in cond['pending']:
                if case['reset_index']==5:cases.append(dict(case,condition=cond['condition']))
        assert len(cases)==78
        plan={'round_id':c.name,'gpu_uuid':gpu,'code_commit':subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip(),'inventory_sha256':digest(inv),'cases':cases,'profiles':{m:digest(root/f'profiles/20261109-four-suite-reset0-v4/{m}.pt') for m in ['awq','smoothquant']},'stop_new_remaining_percent':15,'closure_remaining_percent':10,'heartbeat_minutes':15,'measurement_only':True,'after_completion':'strict paired audit, visual report, private incremental backup, Git sync, native shutdown; no automatic precision changes'}
        (c/'plan.json').write_text(json.dumps(plan,indent=2));print('PREPARED',len(cases));return
    plan=json.loads((c/'plan.json').read_text());assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==plan['code_commit']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',MUJOCO_GL='egl',OMP_NUM_THREADS='4',PYTHONPATH=os.pathsep.join(map(str,[code,root/'sources/openvla-oft',root/'sources/LIBERO'])))
    state={'round_id':c.name,'status':'RUNNING','completed':[],'plan_sha256':digest(c/'plan.json')}
    def save():
        state['observed_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=c/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(c/'status.json')
    try:
        for i,case in enumerate(plan['cases']):
            if (c/'STOP_NEW_GPU').exists():state['status']='STOPPED_BY_CONTROL';break
            assert shutil.disk_usage(root).free>=6*1024**3
            assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==plan['gpu_uuid']
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            method=case['condition'].split('_w')[0];bits=case['condition'].split('a')[-1];profile=root/f'profiles/20261109-four-suite-reset0-v4/{method}.pt';assert digest(profile)==plan['profiles'][method]
            name=f"{c.name}-{case['condition']}-{case['suite']}-task{case['task_id']}-reset5";output=root/'runs'/name;assert not output.exists()
            cmd=[str(root/'envs/oft/bin/python'),'-u','-m','qvla.execute_oft_quant_drift_v2','--oft-source',str(root/'sources/openvla-oft'),'--checkpoint',str(root/'models/openvla_oft'),'--suite',case['suite'],'--task-id',str(case['task_id']),'--initial-state-index','5','--model-seed','7','--round-id',name,'--output',str(output),'--profile',str(profile),'--official-root',str(root/'sources/official-quantization-pinned-v2'),'--method',method,'--activation-bits',bits]
            state.update(current_case=case,current_index=i,argv=cmd);save()
            with (root/'logs'/(name+'.log')).open('x') as log:
                proc=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT);state['pid']=proc.pid;save();exitcode=proc.wait()
            assert exitcode==0,('episode process failed',exitcode)
            result=json.loads((output/'result.json').read_text());assert result['gate']=='PASS_ACTUAL_OFT_QUANT_NATURAL_TRACE_PENDING_PAIRED_AUDIT'
            state['completed'].append({'case':case,'result':str(output/'result.json'),'sha256':digest(output/'result.json')});save()
        else:state['status']='COMPLETE_PENDING_PAIRED_AUDIT_AND_CLOSURE'
    except BaseException as e:
        state.update(status='FAILED_STOPPED',error_type=type(e).__name__,error=str(e));raise
    finally:save()

if __name__=='__main__':main()
