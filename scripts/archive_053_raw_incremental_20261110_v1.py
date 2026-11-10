"""053 stopped-round incremental archive; parameters remain server-only by user choice."""
import argparse,datetime,hashlib,io,json,re,subprocess,tarfile
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.root.resolve();out=a.output.resolve();assert out.is_relative_to(r/'archives') and not out.exists();assert subprocess.check_output(['hostname'],text=True).strip()=='autodl-container-c49843b5ba-28bcee5c';assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU must be stopped before final archive';out.mkdir(parents=True)
 selected=[]
 # All today's raw runs repair the earlier incomplete cloud upload too.
 for folder in list((r/'runs').glob('20261110-*'))+list((r/'control').glob('20261110-*'))+[r/'method-code']+list((r/'analysis').glob('20261110-*')):
  if not folder.exists():continue
  for file in folder.rglob('*'):
   if not file.is_file():continue
   assert not file.is_symlink() and file.resolve().is_relative_to(r)
   if file.suffix in ['.pt','.bin','.safetensors','.gguf','.b64','.pyc'] or '__pycache__' in file.parts:continue
   selected.append(file)
 selected+=list((r/'logs').glob('20261110-*.log'))+list((r/'checks').glob('20261110-*'))+list((r/'profiles').glob('20261110-*/*.json'))
 receipt=r/'models/openvla_oft/DOWNLOAD_VERIFICATION.json'
 if receipt.exists():selected.append(receipt)
 for name in ['diagnostics/native_oft_trace_v1.py','diagnostics/oft_multitask_measured_v1.py','qvla/execute_oft_quant_drift_v2.py','qvla/execute_oft_drift_v3.py','qvla/trajectory_diagnostics_v1.py','qvla/on_policy_capture.py']:
  f=r/'code-cpu-prep-20261109-v1'/name
  if f.exists():selected.append(f)
 selected=sorted(set(f for f in selected if f.is_file()));files=[]
 secret=re.compile(rb'-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}')
 for file in selected:
  if file.suffix in ['.py','.json','.jsonl','.log','.md']:
   assert not secret.search(file.read_bytes()),'Credential-shaped content found: '+str(file)
  files.append({'path':str(file.relative_to(r)),'bytes':file.stat().st_size,'sha256':sha(file)})
 parameters=[]
 for file in sorted((r/'profiles').glob('20261110-*/*.pt')):
  parameters.append({'server_path':str(file),'bytes':file.stat().st_size,'sha256':sha(file),'uploaded':False,'downloaded_to_local':False,'reason':'User explicitly retains new quantization parameters on server; cloud parameter copy missing'})
 manifest={'kind':'053_20261110_all_new_raw_runs_plus_code_control_analysis','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'parameter_exclusions':parameters,'private_parent':{'archive_sha256':'7789e201f2339204959d490c3c9fa0e587861fb91a8f781083a2a4f6d7719650','scope':'previous verified82 cases, four reset0 calibration frames, AWQ/SQ profiles; separate private restore index required'},'earlier_today_cloud_gap_repaired_by_this_archive':'all today runs included; earlier 3f3c archive is not required for these raw runs','checkpoint':'models/openvla_oft external public revision638918f3d1c2e43a39a8a20772bdb8b91835e4b7; full download receipt and source dependency indexes retained; not uploaded weights','code_commit':'5a900dd5900fda66f3d3db173a3fc88b73b1fcff prepared code, plus exact method-code files in archive; upstream pins in checks','credential_shape_scan_passed':True,'raw_data_private_only':True}
 data=json.dumps(manifest,indent=2).encode();path=out/'raw-code-results.tar.gz'
 with tarfile.open(path,'w:gz',compresslevel=6) as tar:
  for file in selected:tar.add(file,arcname=str(file.relative_to(r)),recursive=False)
  info=tarfile.TarInfo('MANIFEST.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
 expected={x['path']:x for x in files};verified=0
 with tarfile.open(path,'r:gz') as tar:
  for member in tar:
   h=hashlib.sha256();count=0
   with tar.extractfile(member) as f:
    for b in iter(lambda:f.read(4*1024**2),b''):h.update(b);count+=len(b)
   row={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} if member.name=='MANIFEST.json' else expected[member.name]
   assert count==row['bytes'] and h.hexdigest()==row['sha256'];verified+=1
 assert verified==len(files)+1
 receipt={'gate':'PASS_053_STOPPED_INCREMENTAL_ARCHIVE_MEMBER_SHA','archive':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'members':verified,'members_verified':verified,'files':len(files),'parameter_exclusions':parameters,'security_scan_passed':True,'upload_budget_bytes':2*1024**3,'estimated_network_bytes_if_local_spool_upload_download_verify':3*path.stat().st_size}
 (out/'MANIFEST.json').write_bytes(data);(out/'SERVER_VERIFICATION.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='parameter_exclusions'}))

if __name__=='__main__':main()
