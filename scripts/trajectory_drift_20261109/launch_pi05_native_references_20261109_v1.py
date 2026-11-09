import pathlib,json,subprocess,datetime,hashlib,socket,os
r=pathlib.Path('/root/autodl-tmp/trajectory-drift-20261107');os.chdir(r)
assert socket.gethostname()=='autodl-container-75c54eac1e-02156c9d'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()=='GPU-3e150638-1b75-05b2-0246-a4bd02bb7723'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',MUJOCO_GL='egl',OMP_NUM_THREADS='4',XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONPATH='code:sources/openpi/src:sources/openpi/packages/openpi-client/src:sources/LIBERO')
plans=r/'checks/20261109-pi05-reference-plans-v1';plans.mkdir(exist_ok=False)
for suite in ('libero_spatial','libero_object','libero_goal','libero_10'):
 for task in range(1,10):
  if (r/'control/STOP_NEW_GPU_20261109').exists():raise SystemExit(90)
  assert __import__('shutil').disk_usage(r).free>=6442450944
  rid=f'20261109-216-pi05-{suite}-bf16-task{task}-reset5-v1';out=r/'runs'/rid;plan=plans/(rid+'.json')
  d=dict(round_id=rid,server_id='216',hostname=socket.gethostname(),runner='qvla.execute_pi05_bf16_episode_v1',suite=suite,task_id=task,reset_index=5,model_seed=7,policy_rng_seed=0,condition='bf16',output=str(out),registered_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),role='development shared native BF16 reference, not quantization improvement',primary_remaining_at_series_admission=17)
  plan.write_text(json.dumps(d,indent=2));args=['envs/pi05/bin/python','-u','-m',d['runner'],'--checkpoint','models/pi05_libero','--openpi-source','sources/openpi','--libero-source','sources/LIBERO','--libero-config','libero-config-v1','--plan',str(plan),'--output',str(out),'--round-id',rid,'--suite',suite,'--task-id',str(task),'--reset-index','5','--model-seed','7']
  with (r/'logs'/(rid+'.log')).open('x') as log:rc=subprocess.call(args,env=env,stdout=log,stderr=subprocess.STDOUT)
  with (r/'checks/20261109-216-native-references-status-v1.jsonl').open('a') as f:f.write(json.dumps(dict(round_id=rid,exit_code=rc,plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest()))+'\n')
  if rc:raise SystemExit(rc)
