"""Register fixed-order temporal100 shards against completed gates and exact PTQ references."""
import argparse
import copy
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
    p.add_argument('--offset',type=int,choices=(10,20,30,40),required=True)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--preregistration-commit',required=True)
    p.add_argument('--primary-remaining',type=int,required=True)
    p.add_argument('--primary-observed-at',required=True)
    p.add_argument('--primary-resets-at',type=int,required=True)
    a=p.parse_args();c=a.control;ev=c/'B1-temporal-eval-v2';entry=c/'B1-continuation-v1'
    s=c.parent/'20261016-091-residual-vla-adaptation-v1/source-v1'
    session=Path('/root/autodl-tmp/qvla-repro/backups/experiments/p2-shared-peft/20261016-091-residual-resume-v2')
    r=session/'B1-temporal-v1';phase='reset-%d-%d'%(a.offset,a.offset+9)
    now=datetime.now(timezone.utc)
    if (not 0<=(now-datetime.fromisoformat(a.primary_observed_at)).total_seconds()<=300
            or a.primary_resets_at<=now.timestamp() or a.primary_remaining<=15
            or (c/'STOP_NEW_GPU').exists() or shutil.disk_usage(c).free<8*1024**3):
        raise ValueError('Fresh primary quota above15%,8GiB disk, noSTOP required')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('GPU occupied')
    os.environ['VLA_FROZEN_SOURCE']=str(s);sys.path[:0]=[str(ev),str(s)]
    from run_eval_b1_temporal_091_v2 import parse_args
    from b1_temporal_eval_contract_091_v2 import validate_provenance
    from scripts.vla_stage_supervisor import validate_plan
    baseline=json.loads((ev/'first50-plan-v1/plan.json').read_text())
    prev=r/('eval-v2-first50-gate.json' if a.offset==10 else 'reset-%d-%d-gate.json'%(a.offset-10,a.offset-1))
    gate=json.loads(prev.read_text());expected='PASS_B1_TEMPORAL_PAIRED_PROTOCOL' if a.offset==10 else 'PASS_B1_TEMPORAL_FULL_SHARD_PAIRED_PROTOCOL'
    if gate['gate']!=expected or gate['paired_episodes']!=(50 if a.offset==10 else 100) or gate.get('offset')!=(5 if a.offset==10 else a.offset-10):
        raise ValueError('Previous complete strict gate absent')
    prevplan=ev/'first50-plan-v1/plan.json' if a.offset==10 else entry/('reset-%d-%d'%(a.offset-10,a.offset-1))/'plan.json'
    pp=json.loads(prevplan.read_text());st=json.loads((Path(pp['control_root'])/pp['plan_id']/'status.json').read_text())
    if st['state']!='AWAITING_GATE_REVIEW' or st['last_exit_code']!=0 or len(st['completed_stage_ids'])!=len(pp['stages']):
        raise ValueError('Previous actual stages incomplete')
    if not a.analysis.is_file() or a.analysis.stat().st_size<100:raise ValueError('Analysis absent')
    index=json.loads((c/'COMPLETED_STAGE_BACKUP_INDEX_V2.json').read_text())
    backup=index.get('b1_temporal_v2_first50') if a.offset==10 else index.get('b1_temporal_'+('reset-%d-%d'%(a.offset-10,a.offset-1)))
    if not backup or backup.get('members_verified')!=backup.get('members') or backup.get('gate')!='PASS_LOCAL_RESIDUAL_SESSION_ARCHIVE':
        raise ValueError('Previous complete dual backup not indexed')
    ref=session/phase/'ptq/policy-observations/query-000000.npz'
    event=next(json.loads(x) for x in (session/phase/'ptq/on-policy-events.jsonl').read_text().splitlines() if json.loads(x)['record_type']=='query')
    mg=json.loads((session/(phase+'-gate.json')).read_text())
    if (mg['gate']!='PASS_B3_INIT_ONLY_FULL_SHARD_PAIRED_PROTOCOL' or mg['paired_episodes']!=100
            or event['episode_serial']!=0 or event['query_in_episode']!=0 or sha(ref)!=event['file_sha256']):
        raise ValueError('Matching completed M0 PTQ reference absent/changed')
    mplan=json.loads((c/'M0-continuation-v2'/phase/'plan.json').read_text())
    mcmd=mplan['stages'][0]['command']
    if mcmd[mcmd.index('--initial-state-offset')+1]!=str(a.offset):raise ValueError('PTQ reference wrong reset')
    lock=json.loads((ev/'first50-plan-v1/runtime-lock.json').read_text())
    for f in (prev,a.analysis,prevplan,ref,session/(phase+'-gate.json'),mplan_path(c,phase),entry/'audit_b1_temporal_shard_091_v1.py',Path(__file__),entry/'shard-audit-cpu-test.json'):
        lock['files'][str(f)]=sha(f)
    lock['version']='b1-temporal-continuation-v1-'+phase
    out=entry/phase;lp=out/'runtime-lock.json';stages=[];admission={}
    mutable=('--lock','--output','--initial-state-offset','--num_trials_per_task','--local_log_dir','--ptq-reference-sample','--ptq-reference-sha256')
    for v in ('early80','phase80'):
        old=next(x for x in baseline['stages'] if x['id']=='first50-'+v);stage=copy.deepcopy(old);stage['id']=phase+'-'+v
        cmd=stage['command'];dest=r/phase/v
        for flag,value in (('--lock',str(lp)),('--output',str(dest)),('--initial-state-offset',str(a.offset)),('--num_trials_per_task','10'),('--local_log_dir',str(dest)),('--ptq-reference-sample',str(ref)),('--ptq-reference-sha256',sha(ref))):cmd[cmd.index(flag)+1]=value
        left=list(old['command']);right=list(cmd)
        for flag in mutable:left[left.index(flag)+1]=right[right.index(flag)+1]='<registered-mutable>'
        if left!=right or stage['env']!=old['env'] or stage['cwd']!=old['cwd']:raise ValueError('Uncontrolled GPU field changed')
        args=parse_args(cmd[cmd.index('--candidate'):]);rec=validate_provenance(args)
        admission[v]={'artifact_sha256':args.trained_artifact_sha256,'adapter_sha256':rec['adapter_sha256'],'reference_sha256':sha(ref),'only_allowed_fields_changed':True}
        stages.append(stage)
    env=dict(stages[0]['env']);env['VLA_TEMPORAL_EVAL_SOURCES']=str(ev);py=stages[0]['command'][0]
    stages.append({'id':'audit-'+phase,'cwd':str(s),'env':env,'command':[py,str(s/'scripts/run_locked_peft_stage.py'),'--lock',str(lp),'--output',str(r/'stage-records'/('audit-'+phase)),'--',py,str(entry/'audit_b1_temporal_shard_091_v1.py'),'--root',str(r/phase),'--offset',str(a.offset),'--count','10','--runtime-lock',str(lp),'--output',str(r/(phase+'-gate.json'))]})
    plan={'schema_version':'1.0','plan_id':'20261017-091-B1-temporal80-'+phase+'-v1','control_root':str(out/'control'),'terminal_state':'AWAITING_GATE_REVIEW','stages':stages,'preregistration_commit':a.preregistration_commit,'previous_gate_sha256':sha(prev),'previous_analysis_sha256':sha(a.analysis),'ptq_reference_sha256':sha(ref),'formal_this_shard_per_candidate':100,'formal_total_per_candidate':450,'score_direction_used_for_continuation':False,'primary_remaining_at_registration':a.primary_remaining,'primary_observed_at':a.primary_observed_at,'primary_resets_at':a.primary_resets_at,'GPU_mutable_fields':list(mutable)}
    validate_plan(plan);out.mkdir(parents=True,exist_ok=False)
    bp=out/'previous-backup-admission.json';bp.write_text(json.dumps(backup,indent=2)+'\n');lock['files'][str(bp)]=sha(bp)
    cp=out/'cpu-admission.json';cp.write_text(json.dumps({'gate':'PASS_B1_TEMPORAL_CONTINUATION_CPU','variants':admission,'gpu_used':False},indent=2)+'\n');lock['files'][str(cp)]=sha(cp)
    lp.write_text(json.dumps(lock,indent=2)+'\n');path=out/'plan.json';path.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'gate':'PASS_B1_TEMPORAL_CONTINUATION_REGISTERED','plan':str(path),'sha256':sha(path),'gpu_started':False}))


def mplan_path(c,phase):
    return c/'M0-continuation-v2'/phase/'plan.json'


if __name__=='__main__':main()
