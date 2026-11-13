"""CPU registration only; does not launch GPU or grant a new round authorization.

First restore/verify the completed Spatial raw archive, then register only the
three missing references and four-suite SQ pilots. All snapshots are checked
before any GPU process can be launched through qvla.infra.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from qvla.groot_snapshot_20261111_v2 import SUITES,verify_snapshot
from qvla.infra.common import canonical,digest,write_json,now
from qvla.infra.plan import compile_plan,verify_plan
from qvla.baseline_contract import checkpoint_identity


def completed_spatial(parent):
    source=parent/'libero_spatial'
    if json.loads((source/'reference-gate.json').read_text())['gate']!='PASS_GROOT_REFERENCE_REPEAT_AND_CALIBRATION':
        raise ValueError('Spatial reference gate absent')
    for name in ['bf16','bf16-repeat','calibration-reset0']:
        folder=source/name;report=json.loads((folder/'result.json').read_text())
        if report['gate']!='PASS_GROOT_PAIRED_TRACE' or report['queries']!=len(report['query_files']):
            raise ValueError('Incomplete Spatial source')
        if digest(folder/'rollout.mp4')!=report['video_sha256']:raise ValueError('Spatial video SHA changed')
        for file,expected in report['query_files'].items():
            path=folder/file
            if path.parent!=folder or digest(path)!=expected:raise ValueError('Spatial query SHA/path differs')
    backup=json.loads((parent/'backup-receipt.json').read_text())
    if backup.get('remote_download_sha256_verified') is not True:
        raise ValueError('Restore the missing parent private backup before further GPU expansion')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--parent-round',type=Path,required=True);p.add_argument('--round-id',required=True)
    p.add_argument('--control-root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve(strict=True);parent=a.parent_round.resolve(strict=True)
    current=a.control_root.resolve(strict=True)
    from qvla.infra.lifecycle import quota_decision
    quota=json.loads((current/'quota.json').read_text())
    assert quota_decision(quota,dict(quota_max_age_seconds=1020,stop_expansion_remaining_percent=15,close_remaining_percent=10))=='CONTINUE'
    auth=json.loads((current/'authorization.json').read_text())
    assert auth['shutdown_authorized'] and auth['hostname']==socket.gethostname()
    import subprocess
    actual_gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert actual_gpu==auth['gpu_uuid']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    guard=json.loads((current/'guard-launch-receipt.json').read_text())
    assert str(current).encode() in Path('/proc',str(guard['pid']),'cmdline').read_bytes()
    if socket.gethostname()!='autodl-container-1421458302-7454c1c8':raise ValueError('008 hostname differs; review server identity')
    # Full 29-file preflight before model loading, not lazy checking in suite 2.
    receipts={suite:verify_snapshot(root/'models'/info[0],suite) for suite,info in SUITES.items()}
    completed_spatial(parent)
    rd=current/'stages'/a.round_id
    if rd.exists():raise FileExistsError('Unique round id/output required')
    rd.mkdir(parents=True)
    gate=rd/'cpu-preflight.json'
    write_json(gate,dict(gate='PASS_GROOT_FOUR_SNAPSHOT_AND_PARENT_RAW_CPU',snapshots={s:dict(revision=v['revision'],files=v['files'],bytes=v['bytes']) for s,v in receipts.items()},parent_round=str(parent),spatial_not_rerun=True))
    task=json.loads((ROOT/'Ex_Reports/_tasks/multisuite-trajectory-deviation-v1/task.json').read_text())
    control=dict(server_id='008',persistent_root=str(root),heartbeat_seconds=900,quota_max_age_seconds=1020,
                 start_disk_gib=12,runtime_disk_gib=6,stop_expansion_remaining_percent=15,close_remaining_percent=10,
                 shutdown_provider='user_pause',defer_cloud_sync=True,backup_timeout_seconds=1500,
                 backup_argv=['{python}','{repo}/scripts/infra_archive_run.py','--run-dir','{run_dir}','--prepare-only'],
                 git_sync_argv=['{python}','-c',"raise SystemExit('Consolidated sync belongs to current controller before root closure')"])
    missing=[s for s in SUITES if s!='libero_spatial']
    config=dict(schema_version=1,task=task,model=dict(checkpoint=str(root/'models/groot_spatial'),identity_sha256=checkpoint_identity(root/'models/groot_spatial')['sha256']),
                quantization=dict(method='none',weight_bits=16,activation_bits=16),finetuning=dict(method='none'),data={},
                evaluation=dict(seed=7,env_seed=7,suites=[dict(name=s,role='development',task_ids=[0],reset_offset=5,resets_per_task=1) for s in missing]),
                control=control,binding=dict(id='groot-native-paired-resume-v2',supports_task_subsets=True,quantizers=['none'],strategies=['none'],verified_suites=list(SUITES),
                gate_files=[dict(path=str(gate),sha256=digest(gate),expected_gate='PASS_GROOT_FOUR_SNAPSHOT_AND_PARENT_RAW_CPU')],
                evaluation_argv=['{python}','-m','qvla.groot_drift_20261111_v2','--root',str(root),'--round-dir',str(rd),'--suite','{suite}']),
                candidate_series=dict(method='smoothquant',weight_bits=4,activation_bits=[8,4],scope='116 original GR00T QuantVLA targets',alpha=.5,
                                      variant='unfused original-SQ numeric operators; runtime positive input scale',equivalence_relative_action_mse_max=.001,
                                      calibration='suite-matched reset0 time-uniform max8 observations',actual_reload_and_finite_required=True,
                                      reused_spatial_reference=str(parent),real_integer_kernel=False))
    config['scope']=dict(root_control=str(current),same_scientific_recipe_v2=True,current_gpu_uuid=actual_gpu,old_spatial_reference_not_rerun=True,no_windows_shutdown=True)
    write_json(rd/'config.json',config);plan=compile_plan(config,ROOT,rd)
    for suite in SUITES:
        plan['stages'].append(dict(id='smoothquant-'+suite,cwd=str(ROOT),argv=[sys.executable,'-m','qvla.groot_sq_20261111_v2',
            '--root',str(root),'--round-dir',str(rd),'--suite',suite,'--reference-round',str(parent)]))
    for file in ['qvla/groot_snapshot_20261111_v2.py','qvla/groot_drift_20261111_v2.py','qvla/groot_sq_20261111_v2.py','qvla/execute_groot_bf16_episode_v2.py']:
        plan['source_sha256'][str(ROOT/file)]=digest(ROOT/file)
    for suite in SUITES:
        file=root/'models'/SUITES[suite][0]/'DOWNLOAD_VERIFICATION.json'
        plan['inputs'].append(dict(path=str(file),sha256=digest(file)))
    for file in [parent/'libero_spatial/reference-gate.json',parent/'backup-receipt.json',parent/'libero_spatial/bf16/result.json',parent/'libero_spatial/calibration-reset0/result.json']:
        plan['inputs'].append(dict(path=str(file),sha256=digest(file)))
    plan.pop('plan_sha256');plan['plan_sha256']=canonical(plan);verify_plan(plan);write_json(rd/'plan.json',plan)
    write_json(rd/'REGISTRATION_ONLY.json',dict(gpu_started=False,authorization_created=False,new_fresh_primary_and_explicit_current_authorization_required=True,plan_sha256=plan['plan_sha256'],time_utc=now()))
    print(json.dumps(dict(plan=str(rd/'plan.json'),gpu_started=False)))


if __name__=='__main__':main()
