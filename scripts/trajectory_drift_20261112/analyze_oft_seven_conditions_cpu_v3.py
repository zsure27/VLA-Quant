"""Terminal-only 80-case matched analysis: BF16 plus six admitted quantization conditions.

Reuses audited old400 caches, frozen raw pose math and pre-existing thresholds.
"""
import argparse,json,pathlib,sys,types,hashlib,os
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--control',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--qvla-stage',type=pathlib.Path);a=p.parse_args();r=a.root.resolve();root=a.control.resolve();out=a.output.resolve();os.environ['CUDA_VISIBLE_DEVICES']=''
rd=a.qvla_stage or pathlib.Path(json.loads((root/'CURRENT_RUN.json').read_text())['stage']);status=json.loads((rd/'status.json').read_text());assert status['reason']=='task_commands_complete' and status['state']=='PAUSED_SERVER_LEFT_ON'
admission=json.loads((rd/'paired-audit/result.json').read_text());assert admission['gate']=='PASS_QVLA_RAW_QUERY_ACTION_POSE_VIDEO_AND_PAIRED_CASES' and admission['completed']==admission['planned']
out.mkdir(parents=True,exist_ok=False);legacy=r/'code-cpu-prep-20261109-v1';sys.path[:0]=[str(legacy),str(r/'sources/openvla-oft'),str(r/'sources/LIBERO')]
from qvla.infra.common import digest
from qvla.trajectory_diagnostics_v1 import sustained_windows
prior_cache=r/'analysis/20261112-oft-old480-pair-cache-v1';old_result=json.loads((prior_cache/'result.json').read_text());assert old_result['gate']=='PASS_480_COMPLETED_CASES_400_PAIRED_CPU_CACHE' and len(old_result['metrics'])==400
parent=r/'control/20261112-061-oft-other-configs-v4';qrows=[]
series=json.loads((rd/'series.json').read_text());parents=series.get('prior_qvla_stages',[dict(stage=str(parent/'stages/qvla-pilot-v4'),audit='paired-audit/result.json'),dict(stage=str(parent/'stages/qvla-complete80-v1'),audit='paired-audit/result.json')])+[dict(stage=str(rd),audit='paired-audit/result.json')]
for rec in parents:
 stage=pathlib.Path(rec['stage']);audit=json.loads((stage/rec['audit']).read_text());assert audit['gates_sha256']==admission['gates_sha256']
 for f in stage.glob('libero*-cases.json'):qrows+=json.loads(f.read_text())['completed']
assert len(qrows)==80 and len({(x['case']['suite'],x['case']['task_id'],x['case']['reset_index']) for x in qrows})==80
source=legacy/'diagnostics/native_oft_trace_v1.py';body=source.read_text();snippet="profile = 'awq.pt' if m['condition'].startswith('awq') else 'smoothquant.pt'\n        assert result['profile_sha256'] == sha(Path('profiles/20261109-four-suite-reset0-v4')/profile)";assert body.count(snippet)==1
body=body.replace(snippet,"assert result['profile_sha256'] == expected_qvla_gates_sha");m=types.ModuleType('_frozen_seven_qvla');m.expected_qvla_gates_sha=admission['gates_sha256'];exec(compile(body,str(source)+'[QVLA artifact check only]', 'exec'),m.__dict__)
hashes={}
def cached_sha(p):
 p=pathlib.Path(p);s=p.stat();key=(str(p),s.st_size,s.st_mtime_ns)
 if key not in hashes:hashes[key]=digest(p)
 return hashes[key]
m.sha=cached_sha;metrics=[];traces={};cache=out/'qvla-pair-cache';cache.mkdir()
for row in old_result['metrics']:
 f=prior_cache/'pair-cache'/row['cache'];assert digest(f)==row['cache_sha256'];z=np.load(f,allow_pickle=False);key=(row['suite'],row['task_id'],row['reset_index'],row['condition']);traces[key]={k:z[k].copy() for k in z.files};metrics.append(dict(row))
for row in qrows:
 case=row['case'];f=pathlib.Path(row['result_path']);ref=pathlib.Path(case['reference_result']);assert digest(f)==row['result_sha256'] and digest(ref)==case['reference_sha256'];cand=m.load(f.parent);bf=m.load(ref.parent);t,rot,w=m.compare(cand,bf);n=len(t)
 key=(case['suite'],case['task_id'],case['reset_index'],'qvla_w4a16');trace=dict(translation_mm=t*1000,rotation_deg=rot,translation_vector_mm=(cand['position'][:n]-bf['position'][:n])*1000,rotation_vector_deg=w*180/np.pi);traces[key]=trace
 name=f"{case['suite']}-task{case['task_id']}-reset{case['reset_index']}-qvla.npz";np.savez_compressed(cache/name,**trace)
 metrics.append(dict(suite=case['suite'],task_id=case['task_id'],reset_index=case['reset_index'],condition='qvla_w4a16',success=cand['result']['success'],reference_success=bf['result']['success'],common_steps=n,translation_p95_mm=float(np.percentile(t*1000,95)),rotation_p95_deg=float(np.percentile(rot,95)),first_same_input_chunk_mse=float(np.mean((cand['first_chunk']-bf['first_chunk'])**2)),candidate=str(f.parent),reference=str(ref.parent),cache=name,cache_sha256=digest(cache/name)))
threshold_path=r/'analysis/20261109-oft-pilot-v1/frozen-thresholds.json';thresholds=json.loads(threshold_path.read_text());assert thresholds['translation_m']==0.005 and thresholds['rotation_deg']==2.0 and thresholds['persistence']==5
(out/'frozen-thresholds.json').write_bytes(threshold_path.read_bytes())
for row in metrics:
 key=(row['suite'],row['task_id'],row['reset_index'],row['condition']);trace=traces[key]
 row['translation_windows']=sustained_windows(trace['translation_mm']/1000,thresholds['translation_m'],thresholds['persistence']);row['rotation_windows']=sustained_windows(trace['rotation_deg'],thresholds['rotation_deg'],thresholds['persistence'])
 chunks=[]
 for folder in [row['candidate'],row['reference']]:
  files=list(pathlib.Path(folder).glob('episodes/*/query-00000.npz'));assert len(files)==1
  with np.load(files[0],allow_pickle=False) as z:chunks.append({k:z[k].copy() for k in ['image','wrist_image','state','instruction','raw_policy_chunk']})
 left,right=chunks;assert all(np.array_equal(left[k],right[k]) for k in ['image','wrist_image','state','instruction']);delta=left['raw_policy_chunk']-right['raw_policy_chunk'];assert delta.shape==(8,7) and np.isfinite(delta).all();n=min(8,row['common_steps'])
 row['first_executed_positions']=n;row['first_translation_command_rmse']=float(np.sqrt(np.mean(delta[:n,:3]**2)));row['first_rotation_command_rmse']=float(np.sqrt(np.mean(delta[:n,3:6]**2)))
 row['first_translation_by_position']=np.sqrt(np.mean(delta[:n,:3]**2,axis=1)).tolist();row['first_rotation_by_position']=np.sqrt(np.mean(delta[:n,3:6]**2,axis=1)).tolist()
 row['first_gripper_decision_mismatches']=int(sum(m.process_action(left['raw_policy_chunk'][i].copy(),'openvla')[-1]!=m.process_action(right['raw_policy_chunk'][i].copy(),'openvla')[-1] for i in range(n)))
conditions=['awq_w4a16','gptq_w4a16','omniquant_w4a16','smoothquant_w4a8','smoothquant_w4a4','qvla_w4a16'];expected={(s,t,j) for s in m.SUITES for t in range(10) for j in [5,15]}
for condition in conditions:assert {(x['suite'],x['task_id'],x['reset_index']) for x in metrics if x['condition']==condition}==expected
reference_outcomes={}
for row in metrics:
 key=(row['suite'],row['task_id'],row['reset_index']);assert reference_outcomes.get(key,row['reference_success'])==row['reference_success'];reference_outcomes[key]=row['reference_success']
per_condition={}
for condition in conditions:
 rows=[x for x in metrics if x['condition']==condition];per_condition[condition]=dict(completed=len(rows),success=sum(x['success'] for x in rows),reference_success=sum(x['reference_success'] for x in rows),rescue=sum(x['success'] and not x['reference_success'] for x in rows),break_count=sum(not x['success'] and x['reference_success'] for x in rows),median_pair_translation_p95_mm=float(np.median([x['translation_p95_mm'] for x in rows])),median_pair_rotation_p95_deg=float(np.median([x['rotation_p95_deg'] for x in rows])),translation_window_cases=sum(bool(x['translation_windows']) for x in rows),rotation_window_cases=sum(bool(x['rotation_windows']) for x in rows))
summary=dict(gate='PASS_OFT_7_CONDITIONS_80_MATCHED_CASES_480_PAIRS',unique_complete=560,baseline_completed=80,baseline_success=sum(reference_outcomes.values()),quantized_pairs=480,per_condition=per_condition,metrics=metrics,thresholds=thresholds,threshold_sha256=digest(threshold_path),frozen_raw_math_sha256=digest(source),same_cohort='native OpenVLA-OFT; actual envseed0/modelseed7;8 actions/query; fixed checkpoint/profile; four suites, ten tasks, reset5/15',limits='same40 development tasks, two initial states; no five-initial completion or cross-model claim; closed-loop errors include observation divergence; fixed threshold windows are not geometric change-points or demonstrated compensation targets')
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'qvla-raw-audit.json').write_text(json.dumps(m.AUDITS,indent=2)+'\n')
import plot_oft_seven_conditions_v2 as plotting
baseline_positions={}
for row in metrics:
 key=(row['suite'],row['task_id'],row['reset_index'])
 if key not in baseline_positions:baseline_positions[key]=m.load(pathlib.Path(row['reference']))['position'].copy()*1000
plotting.make(out,metrics,traces,conditions,m.SUITES,thresholds,baseline_positions)
print(json.dumps({k:v for k,v in summary.items() if k not in ['metrics','thresholds']}),flush=True)
