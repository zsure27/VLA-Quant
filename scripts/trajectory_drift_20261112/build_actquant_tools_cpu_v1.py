"""Compile original quantization tools on CPU; no model or GPU execution.

Explicit CUDA architecture permits nvcc host compilation without GPU discovery.
This is not method admission or an OpenVLA runtime equivalence result.
"""
import argparse,hashlib,json,os,pathlib,shutil,signal,stat,subprocess,time,zipfile
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--control',type=pathlib.Path,required=True);a=p.parse_args();r=a.root.resolve();out=a.control/'cpu-preparation/actquant-tools-build-v1';out.mkdir(parents=True,exist_ok=False)
source=r/'sources/ActQuant-original-b64791125070';build=r/'builds/ActQuant-oft-original-v1';build.mkdir(parents=True,exist_ok=False);py=r/'envs/oft/bin/python';archive=source/'vendor/tokenizers-cpp.zip';vendor=source/'vendor/tokenizers-cpp'
assert shutil.disk_usage(r).free>=12*1024**3 and not vendor.exists()
rows=[];total=0
with zipfile.ZipFile(archive) as z:
 for info in z.infolist():
  path=pathlib.Path(info.filename);assert path.parts[0]=='tokenizers-cpp' and '..' not in path.parts and not path.is_absolute() and not stat.S_ISLNK(info.external_attr>>16)
  if info.is_dir():continue
  total+=info.file_size;assert total<=256*1024**2 and info.file_size<=32*1024**2
  p=source/'vendor'/path;p.parent.mkdir(parents=True,exist_ok=True);data=z.read(info);p.write_bytes(data);rows.append(dict(path=info.filename,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
(out/'vendor-extraction.json').write_text(json.dumps(dict(zip_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=rows),indent=2)+'\n')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CMAKE_BUILD_PARALLEL_LEVEL='2',CARGO_BUILD_JOBS='2',OMP_NUM_THREADS='2')
configure=['cmake','-S',str(source),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release','-DGGML_CUDA=ON','-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc','-DCMAKE_CUDA_ARCHITECTURES=89','-DBUILD_PYTHON=OFF','-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_BUILD_SERVER=OFF','-DLLAMA_CURL=OFF','-DLLAMA_BUILD_COMMIT=b64791125070652fe6b554e244fe809c79ef5246']
commands=[configure,['cmake','--build',str(build),'--target','llama-quantize','quantize-vision','--parallel','2']]
try:
 for i,cmd in enumerate(commands):
  (out/('argv-'+str(i)+'.json')).write_text(json.dumps(cmd,indent=2)+'\n');start=time.monotonic()
  with (out/('stage-'+str(i)+'.log')).open('xb') as f:proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  while proc.poll() is None:
   if time.monotonic()-start>1800 or shutil.disk_usage(r).free<8*1024**3 or (a.control/'STOP_NEW_GPU').exists():
    os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=45);raise RuntimeError('Bounded CPU build stopped; no model/GPU started')
   time.sleep(15)
  if proc.returncode:raise RuntimeError('Original CPU build command failed; stage'+str(i)+' exit'+str(proc.returncode))
 outputs=[]
 for name in ['llama-quantize','quantize-vision']:
  f=build/'bin'/name;assert f.is_file();outputs.append(dict(path=str(f),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
 result=dict(gate='PASS_ORIGINAL_ACTQUANT_QUANT_TOOLS_CPU_BUILD_ONLY',source_commit='b64791125070652fe6b554e244fe809c79ef5246',build=str(build),outputs=outputs,gpu_execution=False,model_loaded=False,python_runtime_not_built=True,workers=2)
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
except Exception as e:
 (out/'failure.json').write_text(json.dumps(dict(error=repr(e),build_preserved=str(build),model_loaded=False,gpu_execution=False))+'\n');print(json.dumps(dict(cpu_build_failed=repr(e),logs=str(out))),flush=True);raise
