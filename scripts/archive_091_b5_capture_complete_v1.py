"""Archive only completed B5 capture, labels and selectors; preserve sources."""
import json,os,shutil,subprocess,tarfile,hashlib,socket
from pathlib import Path
ROOT=Path('/root/autodl-tmp/qvla-repro')
BASE=ROOT/'backups/experiments/p5-contextual-routing/20261012-091-b5-pilot-v2'
CONTROL=ROOT/'control/b5-pilot-091-20261012-v2'
DEST=ROOT/'backups/091-b5-capture-complete-20261012-v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 assert socket.gethostname()=='autodl-container-dfbc4bb0ff-1e85dd94'
 gate=json.loads((BASE/'capture-label-fit-content-gate-v1.json').read_text());assert gate['gate']=='PASS_B5_CAPTURE_LABEL_FIT_CONTENT_AUDIT'
 status=CONTROL/'plans/control/20261012-091-b5-pilot-v2-capture-label-fit'
 assert json.loads((status/'status.json').read_text())['state']=='AWAITING_GATE_REVIEW'
 DEST.mkdir();data=DEST/'experiment';data.mkdir()
 for name in ('capture-router_train','capture-router_dev','router_train','router_dev','router_train-teacher','router_train-experts','router_dev-teacher','router_dev-experts','router','task-only'):
  shutil.copytree(BASE/name,data/name,copy_function=os.link)
 for name in ('capture-label-fit-content-gate-v1.json','micro-launch-receipt.json'):os.link(BASE/name,data/name)
 shutil.copytree(status,DEST/'completed-control',copy_function=os.link)
 plans=DEST/'registered-plans';plans.mkdir()
 for p in (CONTROL/'plans').glob('*.json'):os.link(p,plans/p.name)
 meta=ROOT/'control/cpu-audit-20261012'
 for name in ('feature-schema-v2.json','norm-stats-v2.json'):os.link(meta/name,DEST/name)
 repo=Path('/root/VLA-Quant');commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
 subprocess.run(['git','-C',str(repo),'bundle','create',str(DEST/f'VLA-Quant-{commit}.bundle'),'--all'],check=True)
 scope={'server_commit':commit,'execution_commit':'eb196183a3b1e5d012ea6dbaed6ee3297ee9ce5d','data_scope':'Complete B5 capture train0-3/dev4, same-observation teacher/expert labels and 500-step router/task-only. Active micro excluded; not final session archive. No holdout.','source':str(BASE),'source_sha256':gate['source_sha256'],'method':'Hard links of completed immutable outputs; source retained'}
 (DEST/'SERVER_ARCHIVE_SCOPE.json').write_text(json.dumps(scope,indent=2)+'\n')
 rows=[f'{sha(p)}  {p.relative_to(DEST).as_posix()}' for p in sorted(DEST.rglob('*')) if p.is_file()]
 (DEST/'SHA256SUMS.txt').write_text('\n'.join(rows)+'\n')
 (DEST/'RESULTS_SHA256SUMS.txt').write_text('\n'.join(r for r in rows if '  experiment/' in r)+'\n')
 tar=DEST.with_suffix('.tar.gz')
 with tarfile.open(tar,'w:gz') as f:f.add(DEST,arcname=DEST.name)
 print(json.dumps({'archive':str(DEST),'export':str(tar),'sha256':sha(tar),'bytes':tar.stat().st_size,'files':len(rows),'server_commit':commit}))
if __name__=='__main__':main()

