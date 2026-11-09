import pathlib,subprocess,json,tarfile,hashlib
pins=[('IST-DASLab/gptq','2d65066eeb06a5c9ff5184d8cebdf33662c67faf'),('OpenGVLab/OmniQuant','feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14'),('Hsu1023/DuQuant','d56cfc6fe97c34c0eb100fec82fe439865905679'),('AIoT-MLSys-Lab/QuantVLA','9e435e56aac4e743526557e5d9ba6e4fddd6ccca')]
rows=[]
for repo,pin in pins:
 name=repo.split('/')[-1];p=pathlib.Path('sources')/(name+'-archive-v1');p.mkdir(exist_ok=False);a=pathlib.Path('tmp')/(name+'-'+pin+'.tar.gz');url='https://codeload.github.com/'+repo+'/tar.gz/'+pin
 subprocess.run(['curl','--fail','--location','--retry','1','--connect-timeout','15','--max-time','90','--max-filesize','67108864','-o',str(a),url],check=True)
 files={}
 with tarfile.open(a) as t:
  for m in t:
   parts=pathlib.PurePosixPath(m.name).parts
   if not m.isfile() or m.size>2*1024*1024 or '..' in parts or len(parts)<2:continue
   rel=pathlib.Path(*parts[1:])
   if rel.suffix not in ['.py','.md','.txt','.sh','.json']:continue
   body=t.extractfile(m).read();target=p/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(body);files[str(rel)]=hashlib.sha256(body).hexdigest()
 assert files
 rows.append({'repository':repo,'commit_addressed_url':url,'requested_commit':pin,'archive_bytes':a.stat().st_size,'archive_sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'path':str(p),'source_files_sha256':files,'scope':'selected source archive only; not a Git checkout or admitted pi05 implementation'});print('SOURCE_ARCHIVE_READY',repo,len(files),flush=True)
pathlib.Path('checks/20261109-216-quant-source-archives-v1.json').write_text(json.dumps(rows,indent=2))
