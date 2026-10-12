"""Archive completed B5 micro/first50 with baseline videos; active slices excluded."""
import json,os,shutil,subprocess,tarfile,hashlib,socket
from pathlib import Path
ROOT=Path('/root/autodl-tmp/qvla-repro')
BASE=ROOT/'backups/experiments/p5-contextual-routing/20261012-091-b5-pilot-v2'
CONTROL=ROOT/'control/b5-pilot-091-20261012-v2'
DEST=ROOT/'backups/091-b5-first50-complete-20261012-v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 assert socket.gethostname()=='autodl-container-dfbc4bb0ff-1e85dd94'
 assert json.loads((BASE/'first50-all-controls-gate.json').read_text())['gate']=='PASS_B5_PHASE_PROTOCOL'
 DEST.mkdir();data=DEST/'experiment';data.mkdir();controls=DEST/'completed-control';controls.mkdir();videos=[]
 for phase in ('micro','first50'):
  status=CONTROL/'plans/control'/('20261012-091-b5-pilot-v2-'+phase)
  assert json.loads((status/'status.json').read_text())['state']=='AWAITING_GATE_REVIEW'
  shutil.copytree(status,controls/phase,copy_function=os.link)
  shutil.copytree(BASE/phase,data/phase,copy_function=os.link)
  for name in ('BF16','A0'):
   logs=list((BASE/phase/name).glob('EVAL-*.txt'));assert len(logs)==1
   paths=[Path(line.split('Saved rollout MP4 at path ',1)[1].strip()) for line in logs[0].read_text().splitlines() if 'Saved rollout MP4 at path ' in line]
   assert len(paths)==(10 if phase=='micro' else 50)
   target=data/'baseline-videos'/phase/name;target.mkdir(parents=True)
   for p in paths:
    source=(CONTROL/'source'/p).resolve();assert (CONTROL/'source/rollouts').resolve() in source.parents
    os.link(source,target/source.name);videos.append({'source':str(source),'archive':str((target/source.name).relative_to(DEST)),'sha256':sha(source)})
 for name in ('micro-gate.json','micro-diagnostic.json','first50-gate.json','first50-all-controls-gate.json','first50-paired-analysis.json','first50-launch-receipt.json','reset-10-19-v3-launch-receipt.json'):os.link(BASE/name,data/name)
 (DEST/'video-manifest.json').write_text(json.dumps({'videos':videos,'count':len(videos)},indent=2)+'\n')
 repo=Path('/root/VLA-Quant');commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
 subprocess.run(['git','-C',str(repo),'bundle','create',str(DEST/f'VLA-Quant-{commit}.bundle'),'--all'],check=True)
 scope={'server_commit':commit,'execution_commit':'eb196183a3b1e5d012ea6dbaed6ee3297ee9ce5d','scope':'Complete B5 micro and first50 raw observations/actions/manifests, 120 baseline external videos, gates/paired analysis/completed control. Active reset10-19 excluded. Combine prior capture archive for training provenance; not final session archive. Same-task development, no holdout.','method':'Hard links from completed outputs; originals retained'}
 (DEST/'SERVER_ARCHIVE_SCOPE.json').write_text(json.dumps(scope,indent=2)+'\n')
 rows=[f'{sha(p)}  {p.relative_to(DEST).as_posix()}' for p in sorted(DEST.rglob('*')) if p.is_file()]
 (DEST/'SHA256SUMS.txt').write_text('\n'.join(rows)+'\n');(DEST/'RESULTS_SHA256SUMS.txt').write_text('\n'.join(r for r in rows if '  experiment/' in r)+'\n')
 tar=DEST.with_suffix('.tar.gz')
 with tarfile.open(tar,'w:gz') as f:f.add(DEST,arcname=DEST.name)
 print(json.dumps({'archive':str(DEST),'export':str(tar),'sha256':sha(tar),'bytes':tar.stat().st_size,'files':len(rows),'videos':len(videos),'server_commit':commit}))
if __name__=='__main__':main()

