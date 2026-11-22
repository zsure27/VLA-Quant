"""Check unchanged measured trajectory protocol and lossless private probe bytes."""
import argparse,os,json,base64
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--reference-repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import numpy as np
from qvla.pi05_omni_episode_admission_v1 import decode_expected
old=(a.reference_repo/'qvla/execute_pi05_torch_awq_episode_v1.py').read_text();new=(a.repo/'qvla/execute_pi05_torch_omni_episode_v1.py').read_text()
needle='    shape = policy_noise_shape(policy)'
assert old[old.index(needle):].replace('awq','omni').replace('AWQ','OMNI_LWC').replace('omni_w4a16','omniquant_w4a16')==new[new.index(needle):]
old=(a.reference_repo/'qvla/audit_pi05_torch_awq_trace_v1.py').read_text();new=(a.repo/'qvla/audit_pi05_torch_omni_trace_v1.py').read_text()
assert old.replace('awq','omni').replace('AWQ','OMNI_LWC').replace('omni_w4a16','omniquant_w4a16')==new
checks=['trajectory_noise_reset_cadence_actions_video_and_result_body_unchanged','raw_audit_equivalent_after_method_label_substitution']
for dtype in [np.float32,np.float64]:
 value=np.linspace(-3,4,70,dtype=dtype).reshape(10,7);row=dict(dtype=value.dtype.str,shape=[10,7],bytes_base64=base64.b64encode(value.tobytes()).decode())
 restored=decode_expected(json.loads(json.dumps(row)))
 assert restored.dtype==value.dtype and restored.tobytes()==value.tobytes()
 bad=dict(row,bytes_base64=base64.b64encode(value.tobytes()[:-1]).decode())
 try:decode_expected(bad)
 except ValueError:pass
 else:raise AssertionError('Truncated probe admitted')
 checks.append(np.dtype(dtype).name+'_probe_bytes_bitwise_and_truncation_rejected')
a.output.mkdir(exist_ok=False,parents=True);out=dict(gate='PASS_OMNI_LWC_UNCHANGED_TRAJECTORY_BODY_AND_PRIVATE_PROBE_CONTRACT_CPU',checks=checks,full_awq_gpu_profile_not_assumed=True,formal_episodes=0)
(a.output/'result.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
