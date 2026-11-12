"""Bounded, pinned ActQuant source preparation while another model owns GPU.

No model loads, build, calibration, dependency installation or GPU launches.
"""
import argparse,hashlib,json,os,pathlib,shutil,subprocess,tarfile,time,urllib.request
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--control',type=pathlib.Path,required=True);a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=''
r=a.root.resolve();out=a.control/'cpu-preparation/actquant-source-v1';out.mkdir(parents=True,exist_ok=False)
commit='b64791125070652fe6b554e244fe809c79ef5246';expected='c9258f07b08b73f9edbce56bcc58acd00e8d4cb1b6854a9257430a4c56af3230';source=r/'sources'/('ActQuant-original-'+commit[:12]);cache=r/'sources/cache';cache.mkdir(exist_ok=True);archive=cache/('ActQuant-'+commit+'.tar.gz')
assert shutil.disk_usage(r).free>=8*1024**3
try:
 if not archive.exists():
  temp=archive.with_suffix('.download');assert not temp.exists();started=time.monotonic();size=0
  with urllib.request.urlopen('https://codeload.github.com/arashakb/ActQuant/tar.gz/'+commit,timeout=35) as response,temp.open('xb') as f:
   while True:
    if time.monotonic()-started>180:raise TimeoutError('Bounded source download exceeded180s; incomplete bytes retained')
    b=response.read(1024**2)
    if not b:break
    size+=len(b);assert size<=16*1024**2;f.write(b)
  assert size==12310197 and hashlib.sha256(temp.read_bytes()).hexdigest()==expected;temp.rename(archive)
 assert archive.stat().st_size==12310197 and hashlib.sha256(archive.read_bytes()).hexdigest()==expected
 assert not source.exists();source.mkdir();rows=[];total=0;prefix='ActQuant-'+commit+'/'
 allowed=['cmake/','ggml/','src/','common/','include/','tools/','vendor/','gguf-py/']
 with tarfile.open(archive) as t:
  for m in t:
   if not m.name.startswith(prefix):continue
   name=m.name[len(prefix):]
   if not name or not m.isfile():continue
   assert not m.issym() and not m.islnk() and '..' not in pathlib.Path(name).parts and not name.startswith('/')
   if not (name in ['CMakeLists.txt','README.md','LICENSE','.gitmodules'] or any(name.startswith(x) for x in allowed)):continue
   assert m.size<=32*1024**2;total+=m.size;assert total<=256*1024**2
   data=t.extractfile(m).read();p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);rows.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
 assert (source/'tools/openvla-oft/CMakeLists.txt').exists()
 result=dict(gate='PASS_PINNED_ACTQUANT_SOURCE_CPU_ONLY',repository='arashakb/ActQuant',commit=commit,archive_bytes=12310197,archive_sha256=expected,source=str(source),selected_bytes=total,files=rows,tools={x:shutil.which(x) for x in ['cmake','g++','cargo','unzip']},cuda_compilers=[str(x) for x in pathlib.Path('/usr/local').glob('cuda*/bin/nvcc')],vendor_zip=(source/'vendor/tokenizers-cpp.zip').exists(),models_loaded=False,gpu_started=False,network_budget_bytes=16*1024**2)
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True)
except Exception as e:
 (out/'failure.json').write_text(json.dumps(dict(error=repr(e),no_model_or_gpu_started=True,partial_sources_retained=True))+'\n');raise
