"""Verify final documents upload, then run archive independently of SSH."""
import hashlib,json,socket,subprocess,tarfile,datetime
from pathlib import Path
ROOT=Path('/root/autodl-tmp/qvla-repro');CONTROL=ROOT/'control'
assert socket.gethostname()=='autodl-container-dfbc4bb0ff-1e85dd94'
packet=CONTROL/'091-final-closing-documents-20261013-v1.tar.gz'
assert packet.stat().st_size==47962327
assert hashlib.sha256(packet.read_bytes()).hexdigest()=='75fa19c33eeeaa8079163d698dae6cca426cc60bc4149588488a28c838552580'
incoming=CONTROL/'091-final-closing-documents-20261013-v1'
assert not incoming.exists()
with tarfile.open(packet,'r:gz') as archive:
 for member in archive.getmembers():
  target=(CONTROL/member.name).resolve();assert incoming==target or incoming in target.parents
  assert member.isfile() or member.isdir(), 'Unexpected archive link/device'
 archive.extractall(CONTROL)
index=json.loads((incoming/'CLOSING_INPUT_INDEX.json').read_text())
assert index['git']=='fb92de80b0708b4d93ed3ea6c54a739fae47e523'
for row in index['files']:
 p=incoming/row['path'];assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
receipt=CONTROL/'091-full-final-archive-runner-v1.json';assert not receipt.exists()
assert not (ROOT/'backups/091-gpu-b5-final-20261013-v1').exists()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
log=CONTROL/'091-full-final-archive-v1.log'
with log.open('x') as stream:
 p=subprocess.Popen(['/root/miniconda3/envs/qvla-oft/bin/python',str(incoming/'archive_091_gpu_full_session_v1.py')],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
value={'gate':'PASS_FINAL_DOCUMENT_SHA_AND_ARCHIVE_RUNNER_LAUNCHED','time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':p.pid,'log':str(log),'incoming_git':index['git'],'files_verified':len(index['files']),'gpu_idle':True,'no_new_gpu':True}
with receipt.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps(value))
