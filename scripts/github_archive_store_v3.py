"""Isolated 256-MiB transport for long uploads; original file identities unchanged.

The original v2 receipt is never merged with this different chunk layout.
All source, private-repo, digest, download-SHA and durable receipt gates reuse
v2 verbatim. Only this invocation's parallel pool limit is 64 (1-MiB buffers).
"""
import argparse,importlib.util,json,sys,threading
from pathlib import Path
SPEC=importlib.util.spec_from_file_location('_private_archive_core_v3',Path(__file__).with_name('github_archive_store_v2.py'))
core=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=core;SPEC.loader.exec_module(core)
core.MAX_WORKERS=64
CHUNK=256*1024**2

class FailFastGitHub(core.GitHub):
 def __init__(self, token):
  self.failed=threading.Event()
  super().__init__(token)
 def ensure_asset(self, part):
  if self.failed.is_set():raise core.ArchiveError('Stopped queued upload after an earlier transfer failure')
  try:return super().ensure_asset(part)
  except Exception:
   self.failed.set();raise
 def verify_download(self, asset, part):
  if self.failed.is_set():raise core.ArchiveError('Stopped verification after an earlier transfer failure')
  try:return super().verify_download(asset,part)
  except Exception:
   self.failed.set();raise

def validate_receipt_layout(state, parts):
 expected={p.whole_sha256:{x.name for x in parts if x.whole_sha256==p.whole_sha256} for p in parts}
 for sha,f in state['files'].items():
  if sha not in expected or any(p['name'] not in expected[sha] for p in f['parts']):
   raise core.ArchiveError('Receipt has a different chunk layout; use a separate receipt')
 return expected

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--plan',type=Path,required=True)
 ap.add_argument('--receipt',type=Path,required=True)
 ap.add_argument('--workers',type=int,default=16,choices=range(1,65))
 a=ap.parse_args();plan=json.loads(a.plan.read_text('utf-8-sig'))
 parts=core.prepare_parts(plan,chunk_size=CHUNK)
 client=FailFastGitHub(core.credential_token());receipt=core.Receipt(a.receipt,client.release)
 original_report=core.report
 def report(**event):
  if client.failed.is_set() and ('upload' in event or 'asset' in event):
   raise core.ArchiveError('Stopped an in-flight upload after an earlier transfer failure')
  original_report(**event)
 core.report=report
 # A receipt with another chunk layout must not be silently repurposed.
 expected=validate_receipt_layout(receipt.state,parts)
 core.run_parts(parts,client,receipt,workers=a.workers)
 core.report(complete=True,archives=len(expected),parts=len(parts),chunk_bytes=CHUNK,workers=a.workers)
if __name__=='__main__':main()
