"""Fetch bounded pinned public source only, verify every saved Git blob."""
import argparse,hashlib,json,pathlib,tarfile,urllib.request
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--trees',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--reuse-archives',type=pathlib.Path);a=p.parse_args()
r,out=a.root.resolve(),a.output.resolve();out.mkdir(parents=True,exist_ok=False);receipts=[]
extensions={'.py','.cpp','.c','.h','.hpp','.cu','.cuh','.sh','.cmake','.md','.json','.toml','.txt','.yaml','.yml','.in','.cl','.comp','.metal'}
for pin in json.loads(a.trees.read_text()):
 repo,commit=pin['repository'],pin['commit'];name=repo.split('/')[-1];dest=out/name;dest.mkdir()
 archive=(a.reuse_archives/(name+'.tar.gz')) if a.reuse_archives and (a.reuse_archives/(name+'.tar.gz')).exists() else out/(name+'.tar.gz');url=f'https://codeload.github.com/{repo}/tar.gz/{commit}';h=hashlib.sha256();total=0
 if not archive.exists():
  with urllib.request.urlopen(url,timeout=90) as response,archive.open('xb') as f:
   while True:
    b=response.read(1024**2)
    if not b:break
    total+=len(b)
    if total>32*1024**2:raise RuntimeError('32MiB compressed source budget exceeded; partial retained')
    f.write(b);h.update(b)
 else:
  with archive.open('rb') as f:
   for b in iter(lambda:f.read(1024**2),b''):total+=len(b);h.update(b)
  assert total<=32*1024**2
 expected={x['path']:x for x in pin['files']};rows=[];selected=0
 with tarfile.open(archive,'r:gz') as t:
  for m in t:
   rel=pathlib.PurePosixPath(m.name);parts=rel.parts[1:]
   if not m.isfile() or not parts or '..' in parts:continue
   path=pathlib.PurePosixPath(*parts);key=path.as_posix()
   if path.suffix not in extensions and path.name not in ['LICENSE','CMakeLists.txt','Makefile','.gitmodules']:continue
   row=expected[key]
   if row['mode']!='100644' and row['mode']!='100755':continue
   assert m.size==row['bytes'] and m.size<=8*1024**2
   selected+=m.size;assert selected<=64*1024**2
   body=t.extractfile(m).read();blob=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest();assert blob==row['git_blob'],key
   target=dest.joinpath(*parts);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(body)
   rows.append(dict(path=key,bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),git_blob=blob))
 receipts.append(dict(repository=repo,commit=commit,url=url,archive_bytes=total,archive_sha256=h.hexdigest(),selected_source_bytes=selected,files=rows,executed_upstream_code=False,compiled=False,model_admitted=False))
 (out/'source-verification.json').write_text(json.dumps(receipts,indent=2)+'\n')
print(json.dumps([dict(repository=x['repository'],compressed_bytes=x['archive_bytes'],source_files=len(x['files']),compiled=False) for x in receipts]))
