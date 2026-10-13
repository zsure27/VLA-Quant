"""Resume verified prefix via bounded SSH ranges; append only validated bytes."""
import hashlib,json,os,shlex,subprocess,time
from pathlib import Path
ROOT=Path.cwd();BASE=ROOT/'backups/experiments/p5-contextual-routing/20261012-091-b5-pilot-v2'
PACKET=BASE/'091-gpu-b5-final-20261013-v1.tar.gz';TOTAL=4201438552
EXPECTED='31ae496b9e14ba7573a703c7123cc8b690e5ffbde33c9b99fed965fdee8121ad'
REMOTE='/root/autodl-tmp/qvla-repro/backups/091-gpu-b5-final-20261013-v1.tar.gz'
SSH=['ssh','-T','-p','18575','-i','C:/Users/zsure/.ssh/id_ed25519_vla_014','-o','UserKnownHostsFile=C:/Users/zsure/AppData/Local/Temp/vla-known-hosts-091-audit-20261012','-o','StrictHostKeyChecking=yes','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','root@connect.nmb1.seetacloud.com']
PY='/root/miniconda3/envs/qvla-oft/bin/python'
TMP=Path('C:/Users/zsure/Documents/Triton/.codex-work/091-archive-range-download-v2')
assert not TMP.exists();TMP.mkdir()
JOURNAL=BASE/'full-download-resume-v2.jsonl'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def record(v):
 with JOURNAL.open('a',encoding='utf-8') as f:f.write(json.dumps(v)+'\n');f.flush();os.fsync(f.fileno())
 print(json.dumps(v),flush=True)
with JOURNAL.open('x') as f:pass
prefix=PACKET.stat().st_size;assert 0<prefix<TOTAL
code=f"import hashlib,json; f=open({REMOTE!r},'rb'); h=hashlib.sha256(); n={prefix}; total=n\nwhile n:\n b=f.read(min(n,8388608)); assert b; h.update(b); n-=len(b)\nprint(json.dumps({{'bytes':total,'sha256':h.hexdigest()}}))"
for attempt in range(3):
 try:
  response=subprocess.run(SSH+[PY+' -c '+shlex.quote(code)],capture_output=True,timeout=120,check=True)
  server=json.loads(response.stdout);break
 except (subprocess.SubprocessError,json.JSONDecodeError) as e:
  record({'event':'PREFIX_RPC_RETRY','attempt':attempt+1,'error_type':type(e).__name__})
else:raise RuntimeError('Cannot verify remote prefix')
assert server['bytes']==prefix and server['sha256']==sha(PACKET)
record({'event':'PASS_EXISTING_PREFIX_SHA','bytes':prefix,'sha256':server['sha256'],'source_retained':True})
offset=prefix
while offset<TOTAL:
 count=min(256*1024*1024,TOTAL-offset)
 code=f"import hashlib,json,sys,os; p={REMOTE!r}; assert os.path.getsize(p)=={TOTAL}; f=open(p,'rb'); f.seek({offset}); b=f.read({count}); assert len(b)=={count}; h={{'offset':{offset},'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}}; sys.stdout.buffer.write(json.dumps(h).encode()+b'\\n'); sys.stdout.buffer.write(b); sys.stdout.buffer.flush()"
 for attempt in range(3):
  part=TMP/f'{offset}-{count}-attempt{attempt+1}.bin';error=TMP/f'{offset}-{count}-attempt{attempt+1}.stderr'
  with error.open('xb') as err:
   process=subprocess.Popen(SSH+[PY+' -c '+shlex.quote(code)],stdout=subprocess.PIPE,stderr=err)
   try:
    header=json.loads(process.stdout.readline())
    assert header['offset']==offset and header['bytes']==count
    digest=hashlib.sha256();received=0
    with part.open('xb') as out:
     while received<count:
      b=process.stdout.read(min(8*1024*1024,count-received))
      if not b:raise RuntimeError('Incomplete SSH range')
      out.write(b);digest.update(b);received+=len(b)
    assert process.wait(timeout=45)==0 and received==count and digest.hexdigest()==header['sha256']
    break
   except Exception as e:
    if process.poll() is None:process.kill();process.wait()
    record({'event':'RANGE_RETRY','offset':offset,'attempt':attempt+1,'error_type':type(e).__name__,'partial_retained':str(part)})
 else:raise RuntimeError('Range failed; verified prefix retained')
 assert PACKET.stat().st_size==offset
 with PACKET.open('ab') as out,part.open('rb') as src:
  for b in iter(lambda:src.read(8*1024*1024),b''):out.write(b)
  out.flush();os.fsync(out.fileno())
 offset+=count
 record({'event':'PASS_RANGE_APPENDED','bytes':count,'prefix_bytes':offset,'total_bytes':TOTAL,'sha256':header['sha256'],'part_retained':str(part)})
assert PACKET.stat().st_size==TOTAL and sha(PACKET)==EXPECTED
record({'gate':'PASS_RESUMED_FULL_ARCHIVE_EXPORT_SHA','bytes':TOTAL,'sha256':EXPECTED,'packet':str(PACKET),'failed_scp_prefix_retained_byte_for_byte':True})
