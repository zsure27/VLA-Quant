"""Archive one completed frozen B5 phase, videos, and completed wrapper receipts."""
import argparse,json,os,shutil,subprocess,tarfile,hashlib,socket
from pathlib import Path
ROOT=Path('/root/autodl-tmp/qvla-repro');BASE=ROOT/'backups/experiments/p5-contextual-routing/20261012-091-b5-pilot-v2'
V2=ROOT/'control/b5-pilot-091-20261012-v2';V3=ROOT/'control/b5-pilot-091-20261012-continuation-v3'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--archive-name',required=True);p.add_argument('--combined',required=True);p.add_argument('--next-launch',required=True);a=p.parse_args()
 assert a.phase in ('reset-10-19','reset-20-29','reset-30-39','reset-40-49') and '/' not in a.archive_name and '/' not in a.combined and '/' not in a.next_launch
 assert socket.gethostname()=='autodl-container-dfbc4bb0ff-1e85dd94'
 gate=json.loads((BASE/(a.phase+'-gate.json')).read_text());assert gate['gate']=='PASS_B5_PHASE_PROTOCOL' and gate['count']==10
 status=V3/'plans/control'/('20261012-091-b5-pilot-v2-'+a.phase+'-continuation-v3')
 assert json.loads((status/'status.json').read_text())['state']=='AWAITING_GATE_REVIEW'
 dest=ROOT/'backups'/a.archive_name;dest.mkdir();data=dest/'experiment';data.mkdir()
 shutil.copytree(BASE/a.phase,data/a.phase,copy_function=os.link);shutil.copytree(status,dest/'completed-control',copy_function=os.link)
 videos=[]
 for name in ('BF16','A0'):
  logs=list((BASE/a.phase/name).glob('EVAL-*.txt'));assert len(logs)==1
  paths=[Path(line.split('Saved rollout MP4 at path ',1)[1].strip()) for line in logs[0].read_text().splitlines() if 'Saved rollout MP4 at path ' in line];assert len(paths)==100
  target=data/'baseline-videos'/name;target.mkdir(parents=True)
  for path in paths:
   src=(V2/'source'/path).resolve();assert (V2/'source/rollouts').resolve() in src.parents
   os.link(src,target/src.name);videos.append({'source':str(src),'archive':str((target/src.name).relative_to(dest)),'sha256':sha(src)})
 for name in (a.phase+'-gate.json',a.phase+'-paired-analysis.json',a.combined,a.next_launch):os.link(BASE/name,data/name)
 wrapper_files=0
 planfiles=list((V2/'plans').glob('*plan.json'))+list((V3/'plans').glob('*plan.json'))
 for planfile in planfiles:
  plan=json.loads(planfile.read_text());statusfile=Path(plan['control_root'])/plan['plan_id']/'status.json'
  if not statusfile.is_file():continue
  state=json.loads(statusfile.read_text())
  if state['state']!='AWAITING_GATE_REVIEW' or state['last_exit_code']!=0:continue
  for stage in plan['stages']:
   c=stage['command'];directory=Path(c[c.index('--output')+1]);target=dest/'completed-wrapper-records'/plan['plan_id']/stage['id'];target.mkdir(parents=True)
   for name in ('invocation.json','console.log','exit-code.txt'):
    source=directory/name;assert source.is_file()
    os.link(source,target/name);wrapper_files+=1
 (dest/'video-manifest.json').write_text(json.dumps({'videos':videos,'count':len(videos)},indent=2)+'\n')
 repo=Path('/root/VLA-Quant');commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
 subprocess.run(['git','-C',str(repo),'bundle','create',str(dest/f'VLA-Quant-{commit}.bundle'),'--all'],check=True)
 scope={'server_commit':commit,'execution_commit':'eb196183a3b1e5d012ea6dbaed6ee3297ee9ce5d','phase':a.phase,'scope':'Complete one 100-episode/config development slice, 200 baseline external videos, protocol/paired/cumulative analysis, completed control and wrapper invocation/stdout/exit receipts. Active phase excluded; prior raw stages retained in prior archives. Not final session archive.','wrapper_files':wrapper_files,'method':'Hard links; source retained'}
 (dest/'SERVER_ARCHIVE_SCOPE.json').write_text(json.dumps(scope,indent=2)+'\n')
 rows=[f'{sha(f)}  {f.relative_to(dest).as_posix()}' for f in sorted(dest.rglob('*')) if f.is_file()]
 (dest/'SHA256SUMS.txt').write_text('\n'.join(rows)+'\n');(dest/'RESULTS_SHA256SUMS.txt').write_text('\n'.join(r for r in rows if '  experiment/' in r)+'\n')
 tar=dest.with_suffix('.tar.gz')
 with tarfile.open(tar,'w:gz') as t:t.add(dest,arcname=dest.name)
 print(json.dumps({'archive':str(dest),'export':str(tar),'sha256':sha(tar),'bytes':tar.stat().st_size,'files':len(rows),'videos':len(videos),'wrapper_files':wrapper_files,'server_commit':commit}))
if __name__=='__main__':main()

