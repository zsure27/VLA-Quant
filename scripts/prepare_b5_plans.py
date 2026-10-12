"""CPU-register immutable, gated B5 plans; never start GPU work."""
import argparse,json,os,socket,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qvla.b5_contract import VERSION,sha,save,source_hashes,candidate_contract,require_baseline_cpu
SPEC=ROOT/'configs/experiments/b5_pilot_v1_20261012.json'
def build(materials,session,output,python,hostname):
    from scripts.vla_stage_supervisor import validate_plan
    output.mkdir(parents=True); material_file=output/'materials.json'
    materials=dict(materials);materials['baseline_gpu_receipt']=str(session/'baseline-gpu-gate.json');materials['bank_smoke_receipt']=str(session/'bank-smoke/receipt.json')
    save(material_file,materials)
    env={'PYTHONPATH':':'.join((str(ROOT),str(ROOT/'diagnostics'),materials['oft_root'],materials['libero_root'])),
         'CUDA_VISIBLE_DEVICES':'0','MUJOCO_GL':'egl','MUJOCO_EGL_DEVICE_ID':'0','PYOPENGL_PLATFORM':'egl','PYTHONHASHSEED':'7',
         'WANDB_MODE':'disabled','TOKENIZERS_PARALLELISM':'false','TF_NUM_INTEROP_THREADS':'1','TF_NUM_INTRAOP_THREADS':'1'}
    lock=output/'runtime-lock.json'
    files={str(ROOT/p):digest for p,digest in source_hashes().items()}
    for p in [material_file,SPEC,ROOT/'scripts/prepare_b5_plans.py',ROOT/'scripts/audit_baseline_gpu_bridge.py',ROOT/'scripts/run_locked_peft_stage.py',
        *(ROOT/'qvla'/n for n in ('run_eval_official_quant.py','model_registry.py','extended_peft.py','b4_joint_peft.py','official_quant_adapter.py','awq_block.py','runtime_contract.py','action_jacobian_batch.py','recovery_lora.py')),
        *(Path(materials[k]) for k in ('w2_g128','w2_g64','w4','b3_adapter','b3_artifact','b4_adapter','b4_artifact','baseline_cpu_receipt','feature_schema','norm_stats')),
        Path(materials['student80'])/'manifest.json',Path(materials['oft_root'])/'experiments/robot/libero/run_libero_eval.py',
        Path(materials['oft_root'])/'experiments/robot/openvla_utils.py',Path(materials['checkpoint'])/'dataset_statistics.json']:
        files[str(p)]=sha(p)
    save(lock,{'version':VERSION,'expected_hostname':hostname,'files':files,'max_stage_seconds':14400,'max_plan_seconds':86400})
    def stages():return []
    def add(target,name,command,directory=None):
        target.append({'id':name,'cwd':str(ROOT),'env':env,'command':list(map(str,[python,ROOT/'scripts/run_locked_peft_stage.py','--lock',lock,'--output',directory or session/'stage-records'/name,'--',*command]))})
    def plan(name,ss):
        value={'schema_version':'1.0','plan_id':f'{session.name}-{name}','control_root':str(output/'control'),'terminal_state':'AWAITING_GATE_REVIEW','stages':ss,'question':'CPU-prepared B5: same-task development; protocol gates only','next_plan':'Audit protocol and live quota; do not choose continuation by success direction'}
        validate_plan(value);save(output/(name+'-plan.json'),value);return value
    def baseline(ss,where,offset,count):
        for name in ('BF16','A0'):
            out=session/where/name
            command=[python,ROOT/'qvla/run_eval_official_quant.py','--model-id',name,'--method','awq','--weight-bits','4','--activation-bits','16',
                '--pretrained_checkpoint',materials['checkpoint'],'--profile',materials['w4'],'--official-root',materials['official_root'],'--libero_root',materials['libero_root'],
                '--task_suite_name','libero_spatial','--initial-state-offset',offset,'--num_trials_per_task',count,'--seed',0,'--env-seed',1,'--seed-protocol','paired',
                '--trace-actions','--trace-observations','--awq-scope','none' if name=='BF16' else 'all','--awq-candidate','profile','--local_log_dir',out]
            add(ss,where+'-'+name,command,out)
    def b5(ss,role,where,offset,count,policy):
        command=[python,ROOT/'qvla/run_eval_b5.py','--materials',material_file,'--feature-schema',materials['feature_schema'],'--norm-stats',materials['norm_stats'],
            '--role',role,'--offset',offset,'--count',count,'--policy',policy,'--output',session/where/policy]
        if policy in ('router','task-only'):command+=['--router',session/policy]
        add(ss,where+'-'+policy,command)
    ss=stages();add(ss,'gpu-resource-gate',[python,ROOT/'scripts/check_b5_gpu_start.py','--materials',material_file,'--expected-hostname',hostname,'--output',session/'gpu-start-receipt.json'])
    add(ss,'actual-protocol-observation-bridge',[python,ROOT/'scripts/probe_qvla_baseline_protocol_cpu_20261012.py','--model-dir',materials['checkpoint'],'--oft-root',materials['oft_root'],'--libero-root',materials['libero_root'],'--tasks',10,'--resets',2,'--output',session/'actual-protocol-observation-bridge.json'])
    baseline(ss,'baseline-micro',0,1)
    add(ss,'audit-baseline-micro',[python,ROOT/'scripts/audit_baseline_gpu_bridge.py','--bf16',session/'baseline-micro/BF16','--a0',session/'baseline-micro/A0','--cpu-receipt',materials['baseline_cpu_receipt'],'--runtime-evaluator',Path(materials['oft_root'])/'experiments/robot/libero/run_libero_eval.py','--count',1,'--output',session/'baseline-micro-gate.json'])
    baseline(ss,'baseline-first50',0,5)
    add(ss,'audit-baseline',[python,ROOT/'scripts/audit_baseline_gpu_bridge.py','--bf16',session/'baseline-first50/BF16','--a0',session/'baseline-first50/A0','--cpu-receipt',materials['baseline_cpu_receipt'],'--runtime-evaluator',Path(materials['oft_root'])/'experiments/robot/libero/run_libero_eval.py','--output',session/'baseline-gpu-gate.json'])
    add(ss,'bank-smoke10',[python,ROOT/'scripts/smoke_b5_gpu.py','--materials',material_file,'--output',session/'bank-smoke'])
    plan('baseline-and-bank-smoke',ss)
    ss=stages()
    for role,offset,count in [('router_train',0,4),('router_dev',4,1)]:
        where='capture-'+role
        for policy in ('fixed-e0','fixed-e1'):b5(ss,role,where,offset,count,policy)
        capture=session/role
        add(ss,'join-'+role,[python,ROOT/'scripts/prepare_b5_router_data.py','--captures',session/where/'fixed-e0',session/where/'fixed-e1','--role',role,'--output',capture])
        for phase in ('teacher','experts'):
            command=[python,ROOT/'scripts/label_b5_observations.py','--materials',material_file,'--capture',capture,'--role',role,'--phase',phase,'--output',session/(role+'-'+phase)]
            if phase=='experts':command+=['--teacher',session/(role+'-teacher')]
            add(ss,role+'-'+phase,command)
    for policy in ('router','task-only'):
        command=[python,ROOT/'scripts/train_b5_router.py','--train',session/'router_train','--dev',session/'router_dev','--train-labels',session/'router_train-experts','--dev-labels',session/'router_dev-experts',
            '--norm-stats',materials['norm_stats'],'--feature-schema',materials['feature_schema'],'--output',session/policy]
        if policy=='task-only':command+=['--task-only']
        add(ss,'fit-'+policy,command)
    plan('capture-label-fit',ss)
    policies=('fixed-e0','fixed-e1','router','task-only')
    def audit_command(where,offset,count,require=False):
        return [python,ROOT/'scripts/audit_b5_protocol.py','--runs',*(session/where/p for p in policies),'--references',session/where/'BF16',session/where/'A0','--offset',offset,'--count',count,'--require' if require else '--output',session/(where+'-gate.json')]
    ss=stages();baseline(ss,'micro',5,1)
    for policy in policies:b5(ss,'development_evaluation','micro',5,1,policy)
    add(ss,'audit-micro',audit_command('micro',5,1));plan('micro',ss)
    previous=('micro',5,1)
    for i,(offset,count) in enumerate(((5,5),(10,10),(20,10),(30,10),(40,10))):
        where='first50' if i==0 else f'reset-{offset}-{offset+count-1}'
        ss=stages();add(ss,'verify-prior',audit_command(*previous,require=True));baseline(ss,where,offset,count)
        for policy in policies+(('random',) if i==0 else ()):b5(ss,'development_evaluation',where,offset,count,policy)
        add(ss,'audit-phase',audit_command(where,offset,count));plan(where,ss);previous=(where,offset,count)
    return {'gate':'PASS_B5_IMMUTABLE_PLANS_CPU_REGISTERED','gpu_started':False,'formal_episode_budget':2750,'additional_baseline_episodes':100,'micro_overlap':'excluded from formal totals','spec_sha256':sha(SPEC),'source_sha256':source_hashes(),'pending':'actual CPU environment probe; baseline GPU gate; bank smoke; data content intersections; >=12GiB free before GPU start'}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('materials','session','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--python',required=True);p.add_argument('--expected-hostname',required=True);a=p.parse_args()
    if socket.gethostname()!=a.expected_hostname:raise ValueError('Wrong instance')
    if a.output.exists():raise ValueError('Never overwrite immutable plans')
    m=json.loads(a.materials.read_text());candidate_contract(m);require_baseline_cpu(m)
    receipt=build(m,a.session,a.output,a.python,a.expected_hostname);save(a.output/'registration-receipt.json',receipt);print(json.dumps({'gate':receipt['gate'],'gpu_started':False}))
if __name__=='__main__':main()
