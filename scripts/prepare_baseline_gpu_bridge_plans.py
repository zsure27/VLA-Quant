"""Register two GPU baseline protocols on identical reset0-4; no launch."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qvla.b5_contract import sha,save
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('materials','session','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--python',required=True);p.add_argument('--expected-hostname',required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Never overwrite preregistration')
    m=json.loads(a.materials.read_text());a.output.mkdir(parents=True)
    lock=a.output/'lock.json';files={str(f):sha(f) for f in [a.materials,ROOT/'scripts/prepare_baseline_gpu_bridge_plans.py',ROOT/'scripts/run_locked_peft_stage.py',ROOT/'qvla/run_eval_official_quant.py',ROOT/'scripts/check_b5_gpu_start.py',ROOT/'scripts/audit_baseline_gpu_bridge.py',Path(m['w4']),Path(m['oft_root'])/'experiments/robot/libero/run_libero_eval.py']}
    save(lock,{'version':'p0-091-protocol-bridge-v1','expected_hostname':a.expected_hostname,'files':files,'max_stage_seconds':14400,'max_plan_seconds':86400})
    env={'PYTHONPATH':':'.join((str(ROOT),str(ROOT/'diagnostics'),m['oft_root'],m['libero_root'])),'CUDA_VISIBLE_DEVICES':'0','MUJOCO_GL':'egl','PYOPENGL_PLATFORM':'egl','PYTHONHASHSEED':'7','TF_NUM_INTRAOP_THREADS':'1','TF_NUM_INTEROP_THREADS':'1','WANDB_MODE':'disabled'}
    stages=[]
    def stage(name,command,out):stages.append({'id':name,'cwd':str(ROOT),'env':env,'command':list(map(str,[a.python,ROOT/'scripts/run_locked_peft_stage.py','--lock',lock,'--output',out,'--',*command]))})
    stage('gpu-resources',[a.python,ROOT/'scripts/check_b5_gpu_start.py','--materials',a.materials,'--expected-hostname',a.expected_hostname,'--output',a.session/'resources.json'],a.session/'stage-records/resources')
    for protocol,seed,env_seed in [('paired',0,1),('upstream',7,0)]:
        for name in ('BF16','A0'):
            out=a.session/protocol/name
            command=[a.python,ROOT/'qvla/run_eval_official_quant.py','--model-id',name,'--method','awq','--weight-bits','4','--activation-bits','16','--profile',m['w4'],'--official-root',m['official_root'],'--pretrained_checkpoint',m['checkpoint'],
                     '--libero_root',m['libero_root'],'--task_suite_name','libero_spatial','--initial-state-offset',0,'--num_trials_per_task',5,'--seed',seed,'--env-seed',env_seed,'--seed-protocol',protocol,'--awq-scope','none' if name=='BF16' else 'all','--trace-actions','--trace-observations','--local_log_dir',out]
            stage(protocol+'-'+name,command,out)
    value={'schema_version':'1.0','plan_id':a.session.name+'-gpu-protocol-bridge','control_root':str(a.output/'control'),'terminal_state':'AWAITING_GATE_REVIEW','stages':stages,
           'question':'Protocol attribution diagnostic on 50 matched episodes per case; do not tune to 93%',
           'limits':'upstream environment stream under project wrapper preloading, not pristine paper command; task/reset reuse not independent',
           'next_plan':'Inspect actual command/input hashes, paired rescue/break and failures before B5 admission; no success-direction selection'}
    from scripts.vla_stage_supervisor import validate_plan
    validate_plan(value);save(a.output/'baseline-bridge-plan.json',value)
    print(json.dumps({'gate':'PASS_GPU_BASELINE_BRIDGE_PLAN_CPU_REGISTERED','gpu_started':False,'episodes':200}))
if __name__=='__main__':main()
