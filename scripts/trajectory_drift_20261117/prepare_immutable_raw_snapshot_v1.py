from pathlib import Path
import json,hashlib,shutil,subprocess,sys
r=Path('/root/autodl-tmp/trajectory-drift-20261107');root=r/'control/20261117-084-pi05-resume-v1';old=r/'control/20261116-084-pi05-resume-v2';oldest=r/'control/20261116-084-pi05-resume-v1';area=root/'.closure/final-stable-v1';code=Path(json.loads((root/'authorization.json').read_text())['control_code'])
assert not (root/'shutdown-receipt.json').exists()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
for parent in [root,old,oldest]:
 for d in (parent/'stages').iterdir():
  p=d/'status.json'
  if p.exists():assert json.loads(p.read_text())['state'] not in ['RUNNING','PREFLIGHT']
area.mkdir(exist_ok=False);excluded=[];sources=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def snapshot(parent,prefix):
 for p in sorted(parent.rglob('*')):
  rel=p.relative_to(parent)
  if '.closure' in rel.parts or '__pycache__' in rel.parts:continue
  assert not p.is_symlink()
  if not p.is_file():continue
  target=prefix/rel
  if p.suffix=='.pt':excluded.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),reason='User requested parameters retained on server; not uploaded'));continue
  dest=area/target;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 sources.append(dict(path=str(parent),archive_prefix=str(prefix)))
snapshot(root,Path('.'))
# Previous same-day round has not obtained a private-cloud verification receipt.
# Preserve its raw increment once, excluding all derivative archive trees/parameters.
if not (old/'backup-receipt.json').exists():snapshot(old,Path('unuploaded-parent-20261116-v2'))
else:
 receipt=json.loads((old/'backup-receipt.json').read_text());assert receipt.get('remote_download_sha256_verified');(area/'PARENT_PRIVATE_BACKUP_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
if not (oldest/'backup-receipt.json').exists():snapshot(oldest,Path('unuploaded-parent-20261116-v1'))
else:
 receipt=json.loads((oldest/'backup-receipt.json').read_text());assert receipt.get('remote_download_sha256_verified');(area/'OLDEST_PARENT_PRIVATE_BACKUP_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
# Include the eight pilot raw references needed to recover the first40 pairing.
for name in ['20261111-084-pi05-torch-independent-v2','20261111-084-pi05-smoothquant-a8-v2']:
 parent=r/'rounds'/name
 for sub in ['case-runs','case-plans','audits']:
  if (parent/sub).exists():snapshot(parent/sub,Path('pilot-parent-data')/name/sub)
(area/'PARAMETERS_RETAINED_ON_SERVER.json').write_text(json.dumps(dict(files=excluded),indent=2))
(area/'RECOVERY_SOURCES.json').write_text(json.dumps(dict(sources=sources,external_model_and_profile_parameters_retained_on_server=True),indent=2))
(area/'status.json').write_text(json.dumps(dict(state='FINAL_AVAILABLE_ROUND_IMMUTABLE_RAW_SNAPSHOT',parameters_excluded=len(excluded),raw_data_and_control_preserved=True)))
subprocess.run([sys.executable,str(code/'scripts/infra_archive_run.py'),'--run-dir',str(area),'--prepare-only'],check=True)
plan=json.loads((area/'.closure/upload-plan.json').read_text());proof=dict(gate='PASS_FINAL_IMMUTABLE_RAW_ARCHIVE_MEMBER_SHA',archive=plan['files'][0],parameter_shards_retained=len(excluded),private_upload_deferred_to_round_end=True)
(root/'FINAL_STABLE_ARCHIVE_VERIFIED.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
