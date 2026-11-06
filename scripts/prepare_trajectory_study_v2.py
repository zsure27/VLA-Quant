"""Blocked CPU templates for separated W/A studies, never starts a server."""
from __future__ import annotations
import json
from pathlib import Path


def build(repo,destination):
    repo,destination=Path(repo),Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    task=json.loads((repo/'configs/infra/trajectory_study_v1/study.json').read_text(encoding='utf-8'))['task']
    question='分离权重与激活量化、重参数化及误差耦合，检查四suite多任务中的轨迹偏离、动作块尾部和阶段错拍；允许无拐点。'
    specs=[('bf16','none',16,16),('awq-w4a16','awq',4,16),('awq-w2a16','awq',2,16),
           ('sq-w4a16','smoothquant',4,16),('sq-w4a8','smoothquant',4,8),('sq-w4a4','smoothquant',4,4)]
    for name,method,w,a in specs:
        template='bf16_multisuite.example.json' if method=='none' else ('awq_multisuite.example.json' if method=='awq' else 'smoothquant_multisuite.example.json')
        cfg=json.loads((repo/'configs/infra'/template).read_text(encoding='utf-8'))
        cfg['task']=task
        cfg['diagnostic_question_v2']=question
        cfg['quantization']['weight_bits']=w;cfg['quantization']['activation_bits']=a
        cfg['control']['server_id']='UNASSIGNED_USER_OPENED_SERVER_REQUIRED'
        for suite in cfg['evaluation']['suites']:suite['resets_per_task']=1
        backend='bf16' if method=='none' else method
        if method=='smoothquant':cfg['quantization']['method']='smoothquant_diagnostic_v2'
        for profile in cfg['quantization'].get('profiles',[]):
            # All SQ A16/A8/A4 controls share exactly ONE W4 search/calibration profile.
            profile['path']=profile['path'].replace('w8','w4') if method=='smoothquant' else profile['path'].replace('w4','w'+str(w))
        private='{repo}/.local/trajectory-study-v2/'
        command=['{python}','-m','qvla.run_trajectory_diagnostic_v2','--identity',private+name+'.identity.json',
                 '--admission',private+name+'.admission.json','--capture-root','{output}/capture','--']
        command += ['--pretrained_checkpoint','{checkpoint}','--task_suite_name','{suite}',
                    '--num_trials_per_task','{count}','--local_log_dir','{output}','--seed','7']
        if method=='none':command += ['--initial_state_offset','{offset}','--env_seed','0','--seed_protocol','paired']
        else:
            command += ['--method',method,'--weight-bits',str(w),'--activation-bits',str(a),
                        '--initial-state-offset','{offset}','--env-seed','0','--seed-protocol','paired',
                        '--official-root',cfg['quantization']['official_root']]
            for profile in cfg['quantization']['profiles']:command += ['--profile',profile['path']]
            command += cfg['quantization'].get('recipe_argv',[])
        cfg['binding']={'id':'trajectory-wa-v2-pending-actual-gpu-admission',
            'verified_suites':[r['name'] for r in cfg['evaluation']['suites']],
            'quantizers':[cfg['quantization']['method']],'strategies':['none'],
            'gate_files':[{'path':private+name+'.admission.json','sha256':'0'*64,'expected_gate':'PASS_TRAJECTORY_CAPTURE_ACTUAL_MODEL_V2'}],
            'evaluation_argv':command}
        cfg['diagnostic_note']='BLOCKED TEMPLATE, interface validation only. SQ is conditional on smoothing-only equivalence. Plugin qvla.trajectory_infra_v2 required. No actual GPU verification.'
        (destination/(name+'.example.json')).write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        identity={'data_kind':'measured','model_id':name,'backend':backend,'weight_bits':w,'activation_bits':a,
                  'checkpoint':cfg['model']['checkpoint'],'checkpoint_sha256':'0'*64,
                  'profile_sha256':{p['path']:p['sha256'] for p in cfg['quantization'].get('profiles',[])},
                  'coordinate_id':'REPLACE_WITH_VERIFIED_COORDINATE_SHA','cohort_id':'REPLACE_WITH_ACTUAL_COHORT'}
        (destination/(name+'.identity.example.json')).write_text(json.dumps(identity,indent=2)+'\n',encoding='utf-8')
    catalog=[
        dict(method='AWQ',axis='weight_only',status='existing_adapter_pending_multisuite_gate',notes='Native backend only W2/W4/W8 A16; W3 and AWQ+A QDQ need separate adapters/identities.'),
        dict(method='SmoothQuant',axis='weight_and_activation',status='existing_adapter_smoothing_equivalence_blocked',notes='W4/W8/W16 with A4/A8/A16; full closed loop forbidden until C0 passes.'),
        dict(method='GPTQ',axis='weight_only',status='literature_candidate_no_oft_binding',notes='Useful W3/W2 comparator; no arbitrary registry rename.'),
        dict(method='OmniQuant-no-LET',axis='weight_and_activation',status='literature_candidate_no_oft_binding',notes='Separate clipping/reconstruction optimization cost from inference.'),
        dict(method='OmniQuant-LET',axis='weight_and_activation',status='literature_candidate_no_oft_binding',notes='LEARNED equivalent transformation needs frozen/reload and FP equivalence controls.'),
        dict(method='DuQuant',axis='weight_and_activation',status='literature_candidate_no_oft_binding',notes='Rotation/reordering/reparameterization require model-specific contracts.'),
        dict(method='QuantVLA',axis='vla_specific',status='paper_reference_not_implemented_here',notes='Not the same name/method as QVLA.'),
        dict(method='Omega-QVLA',axis='vla_specific',status='ppt_reference_identity_unresolved',notes='Pin exact paper/repository/version before integration; no compatibility claim.')]
    study={'version':2,'task':task,'question_v2':question,'status':'CPU_PREPARATION_ONLY_NO_SERVER','methods':catalog,
           'closed_loop_stages':[{'id':'W','conditions':['bf16','awq-w4a16','awq-w2a16'],'suites':4,'tasks_per_suite':10,'resets':[5],'episode_cap':120},
                                 {'id':'A','conditions':['sq-w4a16','sq-w4a8','sq-w4a4'],'requires':'SQ_C0_PASS','suites':4,'tasks_per_suite':10,'resets':[5],'episode_cap':120}],
           'offline_only_controls':['SQ-W16A16-C0','SQ-W16A8','SQ-W16A4'],
           'offline_query_budget':80,'offline_selection':'2 predeclared query states per task across all 40 tasks, no success-only filter; paired BF16 reference; distinct calibration role.',
           'maximum_initial_closed_loop_episodes_if_all_admitted':240,
           'confirmation':'At most 2 predeclared surviving candidates + BF16, resets6/7, at most240 additional episodes; no automatic full method-bitwidth grid.',
           'observed_episodes':0,'training':False,'default_action_head':'BF16 protected for every condition',
           'backup':'Only incremental code/config/aggregate/clearly synthetic figures public. Source PPT and measured raw data stay private.'}
    (destination/'study.json').write_text(json.dumps(study,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return study


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    print(json.dumps(build(Path(__file__).resolve().parents[1],args.output),ensure_ascii=False))
