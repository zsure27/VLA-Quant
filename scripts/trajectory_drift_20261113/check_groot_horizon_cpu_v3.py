import os,sys,json,argparse
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();r=args.root;os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['MUJOCO_GL']='egl';os.environ['PYOPENGL_PLATFORM']='egl';sys.path[:0]=[str(r/'sources/Isaac-GR00T'),str(r/'sources/LIBERO')]
from examples.Libero.eval.run_libero_eval import get_libero_env,get_libero_dummy_action
from libero.libero import benchmark
import numpy as np
suite=benchmark.get_benchmark_dict()['libero_10']();task=suite.get_task(0);init=suite.get_task_init_states(0)[5];traces=[]
for horizon,count in [(1000,15),(1010,1010)]:
 from libero.libero import get_libero_path
 from libero.libero.envs.env_wrapper import ControlEnv
 env=ControlEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),has_offscreen_renderer=False,use_camera_obs=False);assert env.env.horizon==1000 and not env.env.ignore_done;env.env.horizon=horizon;env.seed(7);env.reset();obs=env.set_init_state(init);rows=[]
 for i in range(count):
  obs,_,done,_=env.step(get_libero_dummy_action());rows.append(np.asarray(obs['robot0_eef_pos']).copy())
 assert env.env.timestep==count
 if horizon==1010:assert env.env.done
 env.close();traces.append(np.asarray(rows[:15]))
assert np.array_equal(*traces)
out=dict(gate='PASS_GROOT_LONG_HORIZON_CPU',old_internal_horizon=1000,new_internal_horizon=1010,settling_steps=10,model_actions=1000,first15_dummy_states_bitwise_equal=True,exactly1010_steps_executed=True,gpu_used=False,model_loaded=False)
p=args.output;p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
