"""Original published OFT QVLA calibration, guarded by an existing real BF16 query.

Imports the pinned original package in a separate process. Only adds an input /
native action bridge before calibration. No changes to proxy or quantization math.
"""
import argparse,importlib,json,pathlib,sys
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--round-dir',type=pathlib.Path,required=True);p.add_argument('--profile',type=pathlib.Path,required=True);a=p.parse_args()
sys.argv += ['--task_suite_name','libero_spatial']
r,rd=a.root.resolve(),a.round_dir.resolve();source=r/'sources/QVLA-official-oft-20261110-v1/openvla-oft';sys.path.insert(0,str(source))
import qvla.runtime as runtime
original=runtime.load_components
def guarded_load(cfg,utils):
 model,head,proprio,processor=original(cfg,utils)
 z=np.load(r/'runs/20261108-053-oft-bf16-admission-v3/episodes/task-00-reset-005/query-00000.npz',allow_pickle=False)
 obs={'full_image':z['image'].copy(),'wrist_image':z['wrist_image'].copy(),'state':z['state'].copy()}
 actual=np.asarray(utils.get_vla_action(cfg,model,processor,obs,str(z['instruction'].item()),head,proprio))
 expected=z['raw_policy_chunk'];assert actual.shape==(8,7) and np.isfinite(actual).all()
 checks=dict(input_camera_and_proprio_from_original_query=True,zero_quantization_native_actions_byte_equal=bool(np.array_equal(actual,expected)),max_abs=float(np.abs(actual-expected).max()),base_and_heads_frozen=all(not p.requires_grad for m in [model,head,proprio] for p in m.parameters()))
 (rd/'native-first-query-bridge.json').write_text(json.dumps(dict(gate='PASS_QVLA_NATIVE_REAL_BF16_QUERY_BRIDGE' if all([checks['zero_quantization_native_actions_byte_equal'],checks['base_and_heads_frozen']]) else 'FAIL_QVLA_NATIVE_REAL_BF16_QUERY_BRIDGE',checks=checks),indent=2)+'\n')
 assert checks['zero_quantization_native_actions_byte_equal'] and checks['base_and_heads_frozen'],'QVLA native backend does not match frozen original BF16'
 return model,head,proprio,processor
runtime.load_components=guarded_load
from qvla.calibrate_actions import calibrate
args=argparse.Namespace(seed=7,calib_jsonl=rd/'calibration/records.jsonl',max_samples=4,out_dir=a.profile,pretrained_checkpoint=r/'models/openvla_oft',memory_fraction=.95,libero_root=str(r/'sources/LIBERO'),task_suite_name='libero_spatial',reference_checks=4,save_every=1)
calibrate(args)
