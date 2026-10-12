"""Full completed GPU session archive; CPU materials and checkpoint identity retained."""
import datetime,hashlib,json,os,shutil,socket,subprocess,tarfile
from pathlib import Path
ROOT=Path('/root/autodl-tmp/qvla-repro')
BASE=ROOT/'backups/experiments/p5-contextual-routing/20261012-091-b5-pilot-v2'
V2=ROOT/'control/b5-pilot-091-20261012-v2';V3=ROOT/'control/b5-pilot-091-20261012-continuation-v3'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert socket.gethostname()=='autodl-container-dfbc4bb0ff-1e85dd94'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
state=json.loads((V3/'plans/control/20261012-091-b5-pilot-v2-reset-40-49-continuation-v3/status.json').read_text())
assert state['state']=='AWAITING_GATE_REVIEW' and state['last_exit_code']==0 and len(state['completed_stage_ids'])==8
combined=json.loads((BASE/'combined450-paired-analysis.json').read_text())
assert combined['gate']=='PASS_AUDITED_DISJOINT_B5_PAIRED_COMBINED' and combined['episodes']==450
for name,offset,count in [('first50',5,5),('reset-10-19',10,10),('reset-20-29',20,10),('reset-30-39',30,10),('reset-40-49',40,10)]:
 file=BASE/('first50-all-controls-gate.json' if name=='first50' else name+'-gate.json');g=json.loads(file.read_text())
 assert g['gate']=='PASS_B5_PHASE_PROTOCOL' and (g['offset'],g['count'])==(offset,count)
dest=ROOT/'backups/091-gpu-b5-final-20261013-v1';assert not dest.exists()
incoming=ROOT/'control/091-final-closing-documents-20261013-v1';assert incoming.is_dir()
sources={'experiment':BASE,'frozen-control-v2':V2,'continuation-control-v3':V3,'baseline-complete':ROOT/'backups/091-baseline-complete-20261012-v1','baseline-video-increment':ROOT/'backups/091-baseline-videos-20261012-v1','prior-cpu-audit':ROOT/'backups/cpu-audit-091-final-20261012-v1','closing-documents':incoming}
logical=sum(p.stat().st_size for source in sources.values() for p in source.rglob('*') if p.is_file())
free=shutil.disk_usage(ROOT).free;assert free>=logical+6*1024**3, (free,logical)
dest.mkdir()
for label,source in sources.items():
 assert source.is_dir()
 assert not any(p.is_symlink() for p in source.rglob('*')), 'Unexpected archive symlink'
 shutil.copytree(source,dest/label,copy_function=os.link,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
lock=json.loads((V2/'plans/runtime-lock.json').read_text());material=[]
for name,expected in lock['files'].items():
 src=Path(name);assert sha(src)==expected
 target=dest/'locked-materials'/src.relative_to('/root');target.parent.mkdir(parents=True,exist_ok=True);os.link(src,target)
 material.append({'source':name,'archive':str(target.relative_to(dest)),'sha256':expected})
# Copy OFT correction as a complete source overlay, not only two locked entry files.
shutil.copytree(ROOT/'control/cpu-audit-20261012/oft-corrected-v1',dest/'oft-corrected-v1',copy_function=os.link,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
repo=Path('/root/VLA-Quant');commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip();short=subprocess.check_output(['git','-C',str(repo),'rev-parse','--short','HEAD'],text=True).strip()
bundle=dest/f'VLA-Quant-{commit}.bundle';subprocess.run(['git','-C',str(repo),'bundle','create',str(bundle),'--all'],check=True);os.link(bundle,dest/f'VLA-Quant-{short}.bundle')
scope={'server_commit':commit,'execution_commit':'eb196183a3b1e5d012ea6dbaed6ee3297ee9ce5d','time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Full current 091 GPU run: all P5 raw collection/labels/router parameters/micro/2750 formal episodes, complete baseline bridge and raw videos, executing source and all plans/control stdout/exit receipts, CPU audit/data/materials/profiles, all runtime-locked expert weights, OFT overlay, local latest report/code Git bundle and archive index. Sources retained.','checkpoint_scope':'7B base checkpoint is retained on server and bound by full prior checksum/provenance, not copied into this experiment archive; no full-checkpoint local replica claim. LIBERO/OFT external dependencies remain hash/version references. Prior 046/091 full training archives also retained as referenced independent verified archives.','logical_source_bytes':logical,'free_bytes_before':free,'materials':material,'method':'Hard links to completed immutable inputs and raw outputs; no unique data removed'}
(dest/'SERVER_ARCHIVE_SCOPE.json').write_text(json.dumps(scope,indent=2)+'\n')
rows=[f'{sha(f)}  {f.relative_to(dest).as_posix()}' for f in sorted(dest.rglob('*')) if f.is_file()]
(dest/'SHA256SUMS.txt').write_text('\n'.join(rows)+'\n');(dest/'RESULTS_SHA256SUMS.txt').write_text('\n'.join(r for r in rows if '  experiment/' in r)+'\n')
export=dest.with_suffix('.tar.gz')
with tarfile.open(export,'w:gz',compresslevel=1) as t:t.add(dest,arcname=dest.name)
result={'gate':'PASS_FULL_COMPLETED_GPU_SESSION_ARCHIVE_CREATED','archive':str(dest),'export':str(export),'bytes':export.stat().st_size,'sha256':sha(export),'files':len(rows),'data_files':sum('  experiment/' in r for r in rows),'server_commit':commit,'base_checkpoint_full_local_copy':False}
with (BASE/'final-archive-server-receipt.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
