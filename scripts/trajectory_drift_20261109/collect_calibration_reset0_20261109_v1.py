from pathlib import Path
import hashlib,json
import numpy as np
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env,get_libero_image,get_libero_wrist_image,get_libero_dummy_action,quat2axisangle
root=Path('calibration/20261109-four-suite-reset0-v1');root.mkdir(parents=True,exist_ok=False)
rows=[]
for i,s in enumerate(('libero_spatial','libero_object','libero_goal','libero_10')):
 suite=benchmark.get_benchmark_dict()[s](); initial=np.asarray(suite.get_task_init_states(0)[0]); env,description=get_libero_env(suite.get_task(0),'openvla',256)
 try:
  env.reset();obs=env.set_init_state(initial)
  for _ in range(10):
   obs,_,done,_=env.step(get_libero_dummy_action('openvla'))
   if done: raise ValueError('terminal during wait')
  state=np.concatenate((obs['robot0_eef_pos'],quat2axisangle(obs['robot0_eef_quat']),obs['robot0_gripper_qpos']))
  p=root/f'sample-{i:04d}.npz';np.savez_compressed(p,image=get_libero_image(obs),wrist_image=get_libero_wrist_image(obs),state=state,action=np.zeros(7,dtype=np.float32),instruction=np.asarray(description))
  rows.append(dict(suite=s,task_id=0,reset_index=0,role='quant_calibration',action_field='unused_placeholder_not_teacher_label',initial_state_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 finally:env.close()
(root/'manifest.json').write_text(json.dumps(dict(gate='PASS_DISTINCT_RESET0_CALIBRATION_CAPTURE',evaluation_resets=[5,15,25,35,45],samples=rows),indent=2))
print('PASS_DISTINCT_RESET0_CALIBRATION_CAPTURE',len(rows))
