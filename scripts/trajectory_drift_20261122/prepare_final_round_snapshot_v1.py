"""Immutable closure snapshot; live guard/quota files are explicit point-in-time captures."""
from pathlib import Path
import argparse,json,hashlib,datetime,subprocess,shutil,socket,sys

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--control',type=Path,required=True);p.add_argument('--expected-authorization-sha256',required=True);a=p.parse_args();c=a.control.resolve()
 assert sha(c/'authorization.json')==a.expected_authorization_sha256
 auth=json.loads((c/'authorization.json').read_text());assert auth['hostname']==socket.gethostname()
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==auth['gpu_uuid']
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 final=json.loads((c/'analysis-final-pi05-v2/FINAL_GEOMETRY_AUDIT.json').read_text());assert final['gate']=='PASS_ADMITTED_PI05_200_EACH_GEOMETRY_FINALIZATION'
 for receipt in (c/'stages').glob('*/launch-receipt.json'):
  proc=Path('/proc',str(json.loads(receipt.read_text())['supervisor_pid']),'cmdline');assert not proc.exists() or str(receipt.parent).encode() not in proc.read_bytes()
  st=json.loads((receipt.parent/'status.json').read_text());assert st['reason']=='task_commands_complete' and all(x['exit_code']==0 for x in st['stages'])
 area=c/'.closure/final-current-round-snapshot-v1';area.mkdir(parents=True,exist_ok=False);captures=[];retained=[]
 for source in sorted(c.rglob('*')):
  relative=source.relative_to(c)
  if '.closure' in relative.parts or '__pycache__' in relative.parts:continue
  assert not source.is_symlink()
  if not source.is_file():continue
  if source.suffix in ['.pt','.pth','.safetensors','.bin']:
   retained.append(dict(path=str(source),bytes=source.stat().st_size,sha256=sha(source)));continue
  dest=area/'current-control'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
  mutable=source.name in ['quota.json','guard-status.json','gpu-idle-watch.json','guard.log']
  if mutable:
   data=source.read_bytes();dest.write_bytes(data);digest=hashlib.sha256(data).hexdigest()
  else:
   digest=sha(source);shutil.copyfile(source,dest);assert sha(source)==sha(dest)==digest
  captures.append(dict(path=relative.as_posix(),bytes=dest.stat().st_size,sha256=digest,point_in_time_operational_capture=mutable))
 (area/'CAPTURE_INDEX_PRIVATE.json').write_text(json.dumps(dict(captured_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=captures,scope='CURRENT_CONTROL_ROUND_ONLY',external_model_profiles_and_historical_raw_retained_server=True,parameters_retained_server=retained,all_requested_quantizers_completed=False),indent=2))
 (area/'status.json').write_text(json.dumps(dict(state='STOPPED_IMMUTABLE_CLOSURE_SNAPSHOT',all_originals_retained=True)))
 code=Path(auth['control_code']);run=subprocess.run([sys.executable,str(code/'scripts/infra_archive_run.py'),'--run-dir',str(area),'--max-upload-gib','2','--prepare-only'],capture_output=True,text=True,timeout=300)
 if run.returncode:print(run.stderr[-2000:]);raise RuntimeError('Snapshot preparation not verified; originals retained')
 plan=json.loads((area/'.closure/upload-plan.json').read_text());members=json.loads((area/'.closure/members.json').read_text())['members']
 result=dict(gate='PASS_FINAL_CURRENT_ROUND_SERVER_SNAPSHOT',scope='CURRENT_CONTROL_ROUND_ONLY',archive=plan['files'][0],members=len(members),member_sha_roundtrip_verified=True,point_in_time_metadata_files=sum(x['point_in_time_operational_capture'] for x in captures),all_originals_retained=True,private_cloud_verified=False,all_task_raw_backed_up=False)
 (c/'cpu/FINAL_SERVER_SNAPSHOT_VERIFIED.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
