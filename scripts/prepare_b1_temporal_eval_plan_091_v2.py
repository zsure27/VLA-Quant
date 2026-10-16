"""Register temporal micro+first50 with a new trained identity and matching PTQ bridge."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',type=Path,required=True)
    p.add_argument('--preregistration-commit',required=True)
    p.add_argument('--primary-remaining',type=int,required=True)
    p.add_argument('--primary-observed-at',required=True)
    p.add_argument('--primary-resets-at',type=int,required=True)
    a=p.parse_args();c=a.control;b=c/'B1-temporal-v1';ev=c/'B1-temporal-eval-v2'
    src=c.parent/'20261016-091-residual-vla-adaptation-v1/source-v1'
    r=Path('/root/autodl-tmp/qvla-repro/backups/experiments/p2-shared-peft/20261016-091-residual-resume-v2/B1-temporal-v1')
    now=datetime.now(timezone.utc)
    if (not 0<=(now-datetime.fromisoformat(a.primary_observed_at)).total_seconds()<=300
            or a.primary_resets_at<=now.timestamp() or a.primary_remaining<=15 or (c/'STOP_NEW_GPU').exists()
            or shutil.disk_usage(c).free<12*1024**3):raise ValueError('Fresh quota/disk/stop admission failed')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('GPU occupied')
    os.environ['VLA_FROZEN_SOURCE']=str(src);sys.path.insert(0,str(src));sys.path.insert(0,str(ev))
    from run_eval_b1_temporal_091_v2 import parse_args
    from b1_temporal_eval_contract_091_v2 import MODELS,validate_provenance,validate_metadata
    from scripts.vla_stage_supervisor import validate_plan
    train=json.loads((b/'training-plan-v1/plan.json').read_text())
    gate=r/'train-1000-pair-gate.json'
    if json.loads(gate.read_text())['gate']!='PASS_B1_TEMPORAL_TRAIN_PAIR':raise ValueError('Training gate absent')
    reference=c.parent/'20261016-091-residual-vla-adaptation-v1'
    oldr=Path('/root/autodl-tmp/qvla-repro/backups/experiments/p2-shared-peft/20261016-091-residual-vla-adaptation-v1')
    ref=oldr/'first50/ptq/policy-observations/query-000000.npz'
    # Original PTQ's first complete query is task0/reset5; R0 bridge already proved reboot equivalence.
    event=next(json.loads(l) for l in (oldr/'first50/ptq/on-policy-events.jsonl').read_text().splitlines()
               if json.loads(l)['record_type']=='query')
    if event['episode_serial']!=0 or event['query_in_episode']!=0 or sha(ref)!=event['file_sha256']:
        raise ValueError('Canonical first PTQ reference changed')
    env=dict(train['stages'][0]['env']);env['PYTHONPATH']=str(ev)+':'+env['PYTHONPATH']
    lock=json.loads((b/'training-plan-v1/runtime-lock.json').read_text());files=lock['files']
    for f in [*ev.glob('*.py'),gate,c/'B1_TEMPORAL_TRAINING_ANALYSIS_CN.md',c/'B1_TEMPORAL_EVAL_PRE_RUN_CN.md',c/'B1_TEMPORAL_EVAL_V1_FAILURE_AND_V2_CARD_CN.md',ev/'bridge-cpu-test.json',ref]:
        files[str(f)]=sha(f)
    for v in MODELS:
        for f in (r/v/'steps-1000').glob('*'):
            if f.is_file():files[str(f)]=sha(f)
    lock['version']='b1-temporal80-eval-v2-20261017'
    planroot=ev/'first50-plan-v1';planroot.mkdir(exist_ok=False);lp=planroot/'runtime-lock.json'
    lp.write_text(json.dumps(lock,indent=2)+'\n')
    py='/root/miniconda3/envs/qvla-oft/bin/python';stages=[];cpu={}
    def add(name,cmd,out):
        stages.append({'id':name,'cwd':str(src),'env':env,'command':[py,str(src/'scripts/run_locked_peft_stage.py'),
                       '--lock',str(lp),'--output',str(out),'--',*cmd]})
    for phase,count in (('micro',1),('first50',5)):
        for v,mid in MODELS.items():
            m=json.loads((b/(v+'-materials.json')).read_text());t=r/v/'steps-1000';out=r/('eval-v2-'+phase)/v
            cmd=[py,str(ev/'run_eval_b1_temporal_091_v2.py'),'--candidate',v,'--model-id',mid,
                 '--method','awq','--weight-bits','2','--activation-bits','16','--profile',m['w2_g128'],
                 '--awq-primary-group64-profile',m['w2_g64'],'--awq-candidate','w2-attention-no-clip-primary-g64',
                 '--official-root',m['official_root'],'--pretrained_checkpoint',m['checkpoint'],'--libero_root',m['libero_root'],
                 '--task_suite_name','libero_spatial','--initial-state-offset','5','--num_trials_per_task',str(count),
                 '--seed','0','--env-seed','1','--seed-protocol','paired','--trace-actions','--trace-observations',
                 '--local_log_dir',str(out),'--trained-artifact',str(t/'artifact.json'),'--trained-artifact-sha256',sha(t/'artifact.json'),
                 '--adapter-state',str(t/'adapter.pt'),'--recovery-training-manifest',str(Path(m['student80'])/'manifest.json'),
                 '--training-spec',str(b/'b1_temporal80_v1.json'),'--training-pair-gate',str(gate),
                 '--ptq-reference-sample',str(ref),'--ptq-reference-sha256',sha(ref)]
            args=parse_args(cmd[2:]);record=validate_provenance(args)
            bad=dict(record,steps=10)
            try:validate_metadata(bad,v,json.loads(args.recovery_training_manifest.read_text()),sha(args.training_spec))
            except ValueError:pass
            else:raise ValueError('CPU negative smoke-as-full artifact admitted')
            cpu[v]={'artifact_sha256':sha(t/'artifact.json'),'adapter_sha256':sha(t/'adapter.pt'),
                    'wrong_steps_rejected':True,'loaded_state_tensor_contract':True}
            add(phase+'-'+v,cmd,out)
        add('audit-'+phase,[py,str(ev/'audit_b1_temporal_eval_091_v2.py'),'--root',str(r/('eval-v2-'+phase)),
                          '--count',str(count),'--output',str(r/('eval-v2-'+phase+'-gate.json'))],r/'stage-records'/('audit-eval-v2-'+phase))
    (planroot/'cpu-admission.json').write_text(json.dumps({'gate':'PASS_B1_TEMPORAL_EVAL_CPU_CONTRACTS','variants':cpu,'gpu_used':False},indent=2)+'\n')
    files[str(planroot/'cpu-admission.json')]=sha(planroot/'cpu-admission.json');lp.write_text(json.dumps(lock,indent=2)+'\n')
    plan={'schema_version':'1.0','plan_id':'20261017-091-B1-temporal80-micro-first50-v2',
          'control_root':str(planroot/'control'),'terminal_state':'AWAITING_GATE_REVIEW','stages':stages,
          'preregistration_commit':a.preregistration_commit,'this_plan_formal_per_variant':50,'formal_target_per_variant':450,
          'micro_overlap_excluded':True,'primary_remaining_at_registration':a.primary_remaining,
          'primary_observed_at':a.primary_observed_at,'primary_resets_at':a.primary_resets_at,
          'next':'First50 strict gate then all failures/rescue-break/task interval/McNemar/cost and complete dual archive; fresh resource quota, next unique plans only in fixed order'}
    validate_plan(plan);path=planroot/'plan.json';path.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'gate':'PASS_B1_TEMPORAL_EVAL_REGISTERED','sha256':sha(path),'plan':str(path),'gpu_started':False}))


if __name__=='__main__':main()
