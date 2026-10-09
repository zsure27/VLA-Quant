"""Register efficient fresh-pair B4 plans after one complete CPU data audit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from scripts.prepare_b4_joint_plans import build_plans as legacy_build
from scripts.audit_b4_data_scope import audit_materials, sha
from scripts.train_b4_joint_peft import materials_contract


def build_plans(materials,session,output,python,hostname):
    plans=legacy_build(materials,session,output,python,hostname)
    for phase,plan in plans.items():
        plan['plan_id']+= '-pair-v2'
        plan['question']='Within-task B4-B3 fresh paired development; archived A4/BF16/A0 references'
        stages=[]
        for stage in plan['stages']:
            if stage['id'].rsplit('-',1)[-1] in ('A4','BF16','A0'):continue
            stage['command']=[str(ROOT/'scripts/audit_b4_pair_v2.py') if arg==str(ROOT/'scripts/audit_b4_joint_eval.py') else arg
                              for arg in stage['command']]
            stages.append(stage)
        if phase=='first50':
            wrapper=stages[0]['command']; split=wrapper.index('--')
            stages.insert(0,dict(stages[0],id='verify-data-scope',command=[*wrapper[:split+1],python,
                str(ROOT/'scripts/audit_b4_data_scope.py'),'--verify-receipt',str(session/'data-scope-receipt.json')]))
            stages[0]['command'][stages[0]['command'].index('--output')+1]=str(session/'stage-records/verify-data-scope')
        plan['stages']=stages
    return plans


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('materials','session','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--python',required=True);p.add_argument('--expected-hostname',required=True)
    a=p.parse_args()
    if socket.gethostname()!=a.expected_hostname:raise ValueError('Wrong current instance')
    if a.session.exists() or a.output.exists():raise ValueError('Recover immutable existing plan; do not overwrite')
    materials=json.loads(a.materials.read_text(encoding='utf-8'))
    materials_contract(materials,'B4')
    receipt=audit_materials(materials)  # Missing original traces/content evidence => no GPU admission.
    for name in ('backup_active_diagnostics.py','vla_shutdown_remote.py'):
        if sha(Path(materials['closure_tools'])/name)!=sha(ROOT/'scripts'/name):raise ValueError('Sync closure tools first')
    for key in ('oft_root','libero_root','official_root'):
        if not Path(materials[key]).is_dir():raise ValueError('Missing runtime')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('GPU busy; recover current plan')
    if shutil.disk_usage(a.session.parent).free<12*1024**3:raise ValueError('Require 12 GiB free with archive reserve')
    a.session.mkdir(parents=True);a.output.mkdir(parents=True)
    receipt_path=a.session/'data-scope-receipt.json'
    receipt_path.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    plans=build_plans(materials,a.session,a.output,a.python,a.expected_hostname)
    lock_path=a.output/'B4-runtime-lock.json'
    lock=json.loads(lock_path.read_text())
    for path in (receipt_path,ROOT/'scripts/prepare_b4_pair_v2_plans.py',ROOT/'scripts/audit_b4_pair_v2.py',
                 ROOT/'scripts/audit_b4_data_scope.py',ROOT/'configs/experiments/b4_eval_pair_v2_20261009.json'):
        lock['files'][str(path)]=sha(path)
    lock['files'].update(receipt['source_sha256'])
    lock_path.write_text(json.dumps(lock,indent=2)+'\n')
    from scripts.vla_stage_supervisor import validate_plan
    for phase,plan in plans.items():
        validate_plan(plan)
        (a.output/f'B4-{phase}-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'gate':'PASS_B4_PAIR_V2_PLANS_REGISTERED','GPU_started':False,'formal_episodes':400}))


if __name__=='__main__':main()
