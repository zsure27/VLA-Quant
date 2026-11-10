import pathlib,json,hashlib,tarfile,subprocess,re
root=pathlib.Path.cwd();c=root/'control/20261110-053-oft-reset5-resume-v1';analysis=root/'analysis/20261110-oft-reset5-user-close-v2';assert (c/'audit-v2.exit').read_text().strip()=='0'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
a=root/'archives/20261110-053-user-close-v2';a.mkdir(exist_ok=False)
subprocess.run(['git','-C',str(root/'code-cpu-prep-20261109-v1'),'bundle','create',str(a/'source-code.bundle'),'HEAD'],check=True)
parent={'repository':'zsure27/VLA-Quant-Archives','release_tag':'backup-archive-20261017','archive_sha256':'7789e201f2339204959d490c3c9fa0e587861fb91a8f781083a2a4f6d7719650','covers':'previous 82 valid cases, 4 BF16 repeats, raw calibration and AWQ/SQ profiles; profile member SHA independently verified against current CPU receipts','model_source_receipt':json.load(open(root/'models/openvla_oft/DOWNLOAD_VERIFICATION.json'))}
(a/'PARENT_DEPENDENCIES.json').write_text(json.dumps(parent,indent=2))
paths=[]
for p in root.glob('runs/20261110-053-oft-reset5-resume-v1-*'):paths.extend(x for x in p.rglob('*') if x.is_file())
paths.extend(p for p in c.rglob('*') if p.is_file());paths.extend(p for p in analysis.rglob('*') if p.is_file());paths.extend(root.glob('logs/20261110*'));paths.extend([a/'source-code.bundle',a/'PARENT_DEPENDENCIES.json'])
paths=sorted(p for p in set(paths) if p.name!='20261110-closure-archive-v2.log');assert all(not p.is_symlink() for p in paths)
secret=re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def streamsha(f):
 h=hashlib.sha256()
 for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
rows=[]
for p in paths:
 if p.suffix in ['.py','.json','.log','.txt','.sh']:assert not secret.search(p.read_bytes()),str(p)
 rows.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)})
manifest=a/'MEMBERS.json';manifest.write_text(json.dumps(rows,indent=2));archive=a/'20261110-053-user-close-v1.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=3) as t:
 for p in paths:t.add(p,arcname=str(p.relative_to(root)),recursive=False)
 t.add(manifest,arcname='MEMBERS.json',recursive=False)
expected={x['path']:x for x in rows}
with tarfile.open(archive) as t:
 for m in t:
  if m.name=='MEMBERS.json':continue
  assert m.isfile() and m.size==expected[m.name]['bytes'];assert streamsha(t.extractfile(m))==expected[m.name]['sha256']
receipt={'gate':'PASS_INCREMENTAL_ARCHIVE_MEMBER_ROUNDTRIP','archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),'members':len(rows)+1,'members_verified':len(rows)+1,'run_directories':len(list(root.glob('runs/20261110-053-oft-reset5-resume-v1-*'))),'complete':58,'partial_user_interrupted':1,'parent_archive_sha256':parent['archive_sha256'],'security_scan_passed':True}
(a/'SERVER_VERIFICATION.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
public=root/'tmp/20261110-053-public-results.tar.gz'
with tarfile.open(public,'w:gz') as t:
 for p in analysis.iterdir():t.add(p,arcname='measured/'+p.name,recursive=False)
 for name in ['plan.json','status.json','user-close-request.json','analyze_oft_resume_20261110_v2.py','archive_user_close_v1.py']:t.add(c/name,arcname='control/'+name,recursive=False)
 t.add(a/'SERVER_VERIFICATION.json',arcname='SERVER_VERIFICATION.json',recursive=False)
print('PUBLIC',public.stat().st_size,sha(public))
