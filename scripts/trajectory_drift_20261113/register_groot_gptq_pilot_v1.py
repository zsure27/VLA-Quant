"""Register a separately admitted original-core GPTQ pilot; never launches a GPU."""
import argparse,json,sys,socket,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from qvla.infra.common import digest,canonical,write_json
from qvla.infra.plan import compile_plan,verify_plan
from qvla.infra.lifecycle import quota_decision
from qvla.groot_snapshot_20261111_v2 import SUITES,verify_snapshot

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--control-root',type=Path,required=True);p.add_argument('--reference-map',type=Path,required=True);p.add_argument('--template-config',type=Path,required=True);p.add_argument('--round-id',required=True);a=p.parse_args()
 r=a.root.resolve();control=a.control_root.resolve();auth=json.loads((control/'authorization.json').read_text());quota=json.loads((control/'quota.json').read_text())
 assert auth['hostname']==socket.gethostname() and auth['shutdown_authorized']
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert gpu==auth['gpu_uuid']
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 assert quota_decision(quota,dict(quota_max_age_seconds=1020,stop_expansion_remaining_percent=15,close_remaining_percent=10))=='CONTINUE'
 guard=json.loads((control/'guard-launch-receipt.json').read_text());assert str(control).encode() in Path('/proc',str(guard['pid']),'cmdline').read_bytes()
 cpu=control/'cpu-preparation/gptq-small-contract-v3.json';assert json.loads(cpu.read_text())['gate']=='PASS_GROOT_GPTQ_SCALE_CAPTURE_AND_PACK_CPU'
 references=json.loads(a.reference_map.read_text());assert set(references)==set(SUITES)
 inputs=[dict(path=str(cpu),sha256=digest(cpu))]
 for suite,source in references.items():
  source=Path(source);assert json.loads((source/'reference-gate.json').read_text())['gate']=='PASS_GROOT_REFERENCE_REPEAT_AND_CALIBRATION'
  verify_snapshot(r/'models'/SUITES[suite][0],suite)
  for kind in ['bf16','bf16-repeat','calibration-reset0']:
   folder=source/kind;report=json.loads((folder/'result.json').read_text());assert report['gate']=='PASS_GROOT_PAIRED_TRACE'
   assert digest(folder/'rollout.mp4')==report['video_sha256']
   assert set(report['query_files'])=={p.name for p in folder.glob('query-*.npz')}
   for name,sha in report['query_files'].items():assert Path(name).name==name and digest(folder/name)==sha
   inputs.append(dict(path=str(folder/'result.json'),sha256=digest(folder/'result.json')))
 rd=control/'stages'/a.round_id;rd.mkdir(parents=True,exist_ok=False)
 cfg=json.loads(a.template_config.read_text());cfg['quantization']=dict(method='gptq',weight_bits=4,activation_bits=16)
 cfg['evaluation']['suites']=[dict(name=s,role='development',task_ids=[0],reset_offset=5,resets_per_task=1) for s in SUITES]
 cfg['binding'].update(id='groot-original-gptq-fixed-reservoir-v1',quantizers=['gptq'],gate_files=[dict(path=str(cpu),sha256=digest(cpu),expected_gate='PASS_GROOT_GPTQ_SCALE_CAPTURE_AND_PACK_CPU')],evaluation_argv=['{python}','{repo}/scripts/trajectory_drift_20261113/run_groot_gptq_pilot_v1.py','--root',str(r),'--round-dir',str(rd),'--reference-map',str(rd/'reference-map.json'),'--suite','{suite}','--artifact-root',str(r/'artifacts/server-only'/a.round_id)])
 cfg['scope']=dict(root_control=str(control),current_gpu_uuid=gpu,no_windows_shutdown=True,independent_gptq_variant=True)
 cfg['candidate_series']=dict(method='gptq',weight_bits=4,activation_bits=16,group_size=128,block_size=128,damp=.01,actorder=False,calibration='suite reset0, max8 time-uniform observations, fixed BF16 reservoir128 rows/layer',sequential_block_propagation=False,variant_must_be_reported=True,real_integer_kernel=False,derived_parameters_server_only=True,cloud_gap_must_be_listed=True)
 write_json(rd/'reference-map.json',references);write_json(rd/'config.json',cfg);plan=compile_plan(cfg,ROOT,rd);plan['inputs']+=inputs+[dict(path=str(rd/'reference-map.json'),sha256=digest(rd/'reference-map.json'))]
 for f in ['qvla/groot_gptq_trace_20261113_v1.py','scripts/trajectory_drift_20261113/prepare_groot_gptq_v3.py','qvla/pi05_packed_weight_v1.py','qvla/groot_drift_20261113_v3.py','qvla/groot_snapshot_20261111_v2.py','qvla/groot_sq_20261111_v2.py']:
  plan['source_sha256'][str(ROOT/f)]=digest(ROOT/f)
 plan.pop('plan_sha256');plan['plan_sha256']=canonical(plan);verify_plan(plan);write_json(rd/'plan.json',plan);print(json.dumps(dict(plan=str(rd/'plan.json'),gpu_started=False,authorization_created=False)))
if __name__=='__main__':main()
