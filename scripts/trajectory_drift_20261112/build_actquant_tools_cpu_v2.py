"""Resume original tools build, disabling unrelated Pi0.5 Python target only."""
import argparse,json,os,pathlib,subprocess,signal,time,shutil,hashlib
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--control',type=pathlib.Path,required=True);a=p.parse_args();r=a.root.resolve();parent=a.control/'cpu-preparation/actquant-tools-build-v1';out=a.control/'cpu-preparation/actquant-tools-build-v2';out.mkdir(exist_ok=False)
assert (parent/'failure.json').exists();commands=[json.loads((parent/('argv-'+str(i)+'.json')).read_text()) if (parent/('argv-'+str(i)+'.json')).exists() else None for i in [0,1]]
commands[0]+=['-DBUILD_PI05_PYTHON=OFF'];build=r/'builds/ActQuant-oft-original-v1'
if commands[1] is None:commands[1]=['cmake','--build',str(build),'--target','llama-quantize','quantize-vision','--parallel','2']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CMAKE_BUILD_PARALLEL_LEVEL='2',CARGO_BUILD_JOBS='2',OMP_NUM_THREADS='2')
try:
 for i,cmd in enumerate(commands):
  (out/('argv-'+str(i)+'.json')).write_text(json.dumps(cmd,indent=2)+'\n');start=time.monotonic()
  with (out/('stage-'+str(i)+'.log')).open('xb') as f:proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  while proc.poll() is None:
   if time.monotonic()-start>1800 or shutil.disk_usage(r).free<8*1024**3 or (a.control/'STOP_NEW_GPU').exists():os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=45);raise RuntimeError('Bounded CPU preparation stopped')
   time.sleep(15)
  if proc.returncode:raise RuntimeError('Original CPU command stage'+str(i)+' exit'+str(proc.returncode))
 outputs=[]
 for name in ['llama-quantize','quantize-vision']:
  f=build/'bin'/name;assert f.is_file();outputs.append(dict(path=str(f),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
 result=dict(gate='PASS_ORIGINAL_ACTQUANT_QUANT_TOOLS_CPU_BUILD_ONLY',source_commit='b64791125070652fe6b554e244fe809c79ef5246',build=str(build),outputs=outputs,workers=2,gpu_execution=False,model_loaded=False,python_runtime_not_built=True)
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
except Exception as e:
 (out/'failure.json').write_text(json.dumps(dict(error=repr(e),original_sources_unchanged=True,no_gpu_execution=True))+'\n');print(json.dumps(dict(cpu_build_failed=repr(e),logs=str(out))),flush=True);raise
