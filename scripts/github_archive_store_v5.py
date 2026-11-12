"""Same private store/chunk layout with bounded, resumable CDN readback.

Uploads reuse v4's unchanged metadata/digest/source gates. Only interrupted
downloads resume by HTTP Range. Exact Content-Range and final full SHA required.
No original/remote asset deletion; no credentials sent to signed CDN hosts.
"""
import argparse,hashlib,importlib.util,json,pathlib,sys,time,urllib.request,urllib.parse
spec=importlib.util.spec_from_file_location('_private_core_v5',pathlib.Path(__file__).with_name('github_archive_store_v2.py'));core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)
class ResumableReadback(core.GitHub):
 def __init__(self,token,cache):super().__init__(token);self.cache=cache;cache.mkdir(parents=True,exist_ok=True)
 def verify_download(self,asset,part):
  target=self.cache/(part.name+'.readback');received=target.stat().st_size if target.exists() else 0
  if received>part.size:raise core.HashMismatch('Readback cache exceeds exact part length')
  url=None
  for attempt in range(4):
   try:
    if received==part.size:break
    if not received or not url:
     response=self.open_download(asset);url=response.geturl()
     if received:
      response.close();core.safe_url(url,core.DOWNLOAD_HOSTS);response=urllib.request.urlopen(urllib.request.Request(url,headers={'Range':f'bytes={received}-','User-Agent':'qvla-private-archive-readback-v5'}),timeout=90)
    else:
     core.safe_url(url,core.DOWNLOAD_HOSTS);response=urllib.request.urlopen(urllib.request.Request(url,headers={'Range':f'bytes={received}-','User-Agent':'qvla-private-archive-readback-v5'}),timeout=90)
    with response:
     if received:
      expected=f'bytes {received}-{part.size-1}/{part.size}'
      if response.status!=206 or response.headers.get('Content-Range')!=expected:raise core.ArchiveError('CDN did not honor exact remaining range')
     if 'json' in response.headers.get('Content-Type','').lower():raise core.ArchiveError('JSON metadata is not archive bytes')
     if target.is_symlink():raise core.ArchiveError('Readback cache cannot be a symlink')
     mode='ab' if target.exists() else 'xb';last=time.monotonic()
     with target.open(mode) as output:
      while received<part.size:
       b=response.read(min(1024**2,part.size-received))
       if not b:break
       output.write(b);output.flush();received+=len(b)
       if time.monotonic()-last>=45:core.report(download_part=part.index+1,received=received,expected=part.size);last=time.monotonic()
     if received!=part.size:raise ConnectionError('Incomplete CDN stream; exact remaining bytes will resume')
   except (OSError,ConnectionError,TimeoutError) as error:
    if attempt==3:raise
    core.report(download_resume=part.index+1,received=received,expected=part.size,error=type(error).__name__);url=None;time.sleep(2)
  h=hashlib.sha256()
  with target.open('rb') as f:
   for b in iter(lambda:f.read(1024**2),b''):h.update(b)
  if received!=part.size or h.hexdigest()!=part.sha256:raise core.HashMismatch('Resumed full remote download SHA mismatch: '+part.name)

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=pathlib.Path,required=True);p.add_argument('--receipt',type=pathlib.Path,required=True);p.add_argument('--cache',type=pathlib.Path,required=True);p.add_argument('--workers',type=int,choices=[1,2],default=2);a=p.parse_args()
 plan=json.loads(a.plan.read_text());parts=core.prepare_parts(plan,chunk_size=64*1024**2);client=ResumableReadback(core.credential_token(),a.cache);receipt=core.Receipt(a.receipt,client.release);core.run_parts(parts,client,receipt,workers=a.workers);core.report(complete=True,archives=1,parts=len(parts),same_64MiB_layout=True)
if __name__=='__main__':main()
