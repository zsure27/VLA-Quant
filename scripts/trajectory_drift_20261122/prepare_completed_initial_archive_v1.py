"""Snapshot an explicitly selected completed initial while other initial GPU work continues."""
from pathlib import Path
import argparse,json,hashlib,shutil,subprocess,sys,os

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--control',type=Path,required=True);p.add_argument('--initial',type=int,choices=[15],required=True);p.add_argument('--max-archive-bytes',type=int,required=True);a=p.parse_args()
 r=a.root.resolve();c=a.control.resolve();assert c.parent==r/'control';assert 0<a.max_archive_bytes<=270071459
 stages=[c/'stages'/f'20261122-084-duquant-a4-reset{a.initial}-{scope}-v1' for scope in ['pilot4','remaining36']]
 scientific=set();audits=[];sources={};inputs={};retained=[]
 for stage,expected in zip(stages,[4,36]):
  state=json.loads((stage/'status.json').read_text());assert state['reason']=='task_commands_complete' and all(x['exit_code']==0 for x in state['stages'])
  receipt=json.loads((stage/'launch-receipt.json').read_text());proc=Path('/proc',str(receipt['supervisor_pid']),'cmdline');assert not proc.exists() or str(stage).encode() not in proc.read_bytes()
  audit=json.loads((stage/'FULL_RAW_AUDIT.json').read_text());assert audit['gate']=='PASS_DUQUANT_REGISTERED_CASES_FULL_RAW_AUDIT' and audit['complete']==expected and audit['reset_index']==a.initial
  for row in audit['rows']:
   key=(row['suite'],row['task_id'],row['reset_index']);assert key not in scientific;scientific.add(key)
   assert sha(Path(row['result_path']))==row['result_sha256']
  plan=json.loads((stage/'plan.json').read_text());sources.update(plan['source_sha256']);inputs.update({x['path']:x for x in plan['inputs']});audits.append(dict(stage=str(stage),sha256=sha(stage/'FULL_RAW_AUDIT.json'),complete=expected,successes=audit['successes']))
 assert len(scientific)==40
 area=c/'.closure'/f'initial{a.initial}-complete-archive-v1';area.mkdir(parents=True,exist_ok=False)
 def copy(source,dest):
  source=source.resolve();assert source.is_relative_to(r) and source.is_file() and not source.is_symlink()
  h=sha(source)
  if source.suffix in ['.pt','.pth','.safetensors','.bin']:
   retained.append(dict(path=str(source),bytes=source.stat().st_size,sha256=h));return
  dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);assert sha(dest)==h
 for stage in stages:
  for source in sorted(stage.rglob('*')):
   assert not source.is_symlink()
   if source.is_file() and '__pycache__' not in source.parts and '.closure' not in source.parts:copy(source,area/'completed-control'/source.relative_to(c))
 for name,expected in sources.items():
  source=Path(name);assert sha(source)==expected;copy(source,area/'scientific-source'/source.relative_to(r))
 for name in ['authorization.json','CODE_ROOT_LINEAGE_V1.json','PARENT_RECOVERY_INDEX.json','ADMISSION_PHASE_A_RESOLVED_GATE.json','task-manifest.json']:
  copy(c/name,area/'current-provenance'/name)
 parents=[]
 for name,proofname in [('20261121-084-pi05-gptq-continuation-v1','FINAL_INCREMENT_ARCHIVE_VERIFIED.json'),('20261121-084-pi05-post-refresh-v2','FINAL_DELTA_ARCHIVE_VERIFIED_V3.json')]:
  proof=json.loads((r/'control'/name/proofname).read_text());parents.append(proof['archive'])
 (area/'DEPENDENCIES_AND_PARAMETER_GAPS.json').write_text(json.dumps(dict(parent_archives_required=parents,remote_parent_verification_not_assumed=True,external_inputs=list(inputs.values()),parameters_retained_server=retained,other_initials_not_included=True),indent=2))
 (area/'SCIENTIFIC_COVERAGE.json').write_text(json.dumps(dict(condition='duquant_w4a4',initial=a.initial,complete_unique_cases=40,case_keys=sorted(scientific),audits=audits,selected_by_fixed_current_round_initial_order_not_results=True),indent=2))
 (area/'status.json').write_text(json.dumps(dict(state='STOPPED_EXPLICIT_COMPLETED_INITIAL_SNAPSHOT',active_other_initial_not_read=True)))
 code=Path(json.loads((c/'authorization.json').read_text())['control_code'])
 result=subprocess.run([sys.executable,str(code/'scripts/infra_archive_run.py'),'--run-dir',str(area),'--max-upload-gib',str(a.max_archive_bytes/1024**3),'--prepare-only'],capture_output=True,text=True,timeout=240)
 if result.returncode:print(result.stderr[-1800:]);raise RuntimeError('Selected archive not admitted; originals and staging retained')
 plan=json.loads((area/'.closure/upload-plan.json').read_text());proof=dict(gate='PASS_EXPLICIT_COMPLETED_INITIAL_ARCHIVE_MEMBER_SHA',archive=plan['files'][0],complete_cases=40,parents_required=parents,other_running_initial_excluded=True,private_cloud_verified=False)
 (c/'cpu/INITIAL15_ARCHIVE_SERVER_VERIFIED_V1.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof),flush=True)
if __name__=='__main__':main()
