import hashlib,json,pathlib,shutil,subprocess,socket
P=pathlib.Path
root=P('/root/autodl-tmp/qvla-repro')
assert socket.gethostname()=='autodl-container-dfbc4bb0ff-1e85dd94'
dest=root/'backups/cpu-audit-091-final-20261012-v1'
assert not dest.exists()
dest.mkdir()
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(4194304),b''):h.update(c)
 return h.hexdigest()
def copy(src,rel):
 src=P(src); out=dest/rel;out.parent.mkdir(parents=True,exist_ok=True)
 assert src.is_file() and not out.exists()
 shutil.copyfile(src,out)
 assert digest(src)==digest(out)
 return {'source':str(src),'archive':str(rel),'sha256':digest(out),'bytes':out.stat().st_size}
records=[]
cpu=root/'control/cpu-audit-20261012'
for p in sorted(cpu.iterdir()):
 if p.is_file() and p.suffix in ('.json','.log'):
  records.append(copy(p,P('cpu-audit')/p.name))
records.append(copy(cpu/'final-source-v2.tar.gz',P('code/runner-source-eb19618.tar.gz')))
records.append(copy(cpu/'oft-corrected-v1/experiments/robot/libero/run_libero_eval.py',P('runtime/corrected-run_libero_eval.py')))
plans=root/'control/b5-pilot-091-20261012-v2'
for group in ('plans','baseline-bridge-plans'):
 for p in sorted((plans/group).rglob('*.json')):
  records.append(copy(p,P('registration')/group/p.relative_to(plans/group)))
m=json.loads((plans/'plans/materials.json').read_text())
for k in ('w4','w2_g128','w2_g64','b3_adapter','b3_artifact','b4_adapter','b4_artifact'):
 records.append(copy(m[k],P('materials')/(k+P(m[k]).suffix)))
for p in sorted(P(m['student80']).iterdir()):
 if p.is_file():records.append(copy(p,P('materials/student-state80')/p.name))
commit=subprocess.check_output(['git','-C','/root/VLA-Quant','rev-parse','--short','HEAD'],text=True).strip()
bundle=root/'backups/b2b3-046-final-20261009-0001'/('VLA-Quant-'+commit+'.bundle')
records.append(copy(bundle,P(bundle.name)))
for p in sorted((cpu/'closing-documents').rglob('*')):
 if p.is_file():records.append(copy(p,P('closing-documents')/p.relative_to(cpu/'closing-documents')))
scope={'schema_version':1,'server_commit':commit,'executing_code_commit':'eb196183a3b1e5d012ea6dbaed6ee3297ee9ce5d','archive':str(dest),'files':records,
 'prior_complete_archives':[str(root/'backups'/n) for n in ('b2b3-046-final-20261009-0001','b4-091-final-20261010-0001')],
 'public_checkpoint':m['checkpoint'],'checkpoint_hash_receipt':'cpu-audit/actual-tensor-audit-v2.json',
 'excluded':'Public 7B checkpoint remains on persistent disk, all 22 hashes recorded; regenerable SVD spool retained at original path. No unique experiment files removed.',
 'local_prior_archive_verification':'046 13+18494 and 091 8952+8897 per-file SHA checks passed; package SHA8399ed0ff145c1162cf3eca1c9414aa184f9cfe7e08417bef323000e40c78668',
 'gate':'PASS_CURRENT_CPU_ARCHIVE_CREATED_NO_GPU_ADMISSION'}
(dest/'SERVER_ARCHIVE_SCOPE.json').write_text(json.dumps(scope,indent=2)+'\n')
allfiles=[p for p in sorted(dest.rglob('*')) if p.is_file()]
primary=[p for p in allfiles if p.parent==dest]
secondary=[p for p in allfiles if p.parent!=dest]
for name,files in [('SHA256SUMS.txt',primary),('RESULTS_SHA256SUMS.txt',secondary)]:
 (dest/name).write_text(''.join(digest(p)+'  '+p.relative_to(dest).as_posix()+'\n' for p in files))
print(json.dumps({'gate':scope['gate'],'archive':str(dest),'primary_files':len(primary),'result_files':len(secondary),'bytes':sum(p.stat().st_size for p in allfiles)}))
