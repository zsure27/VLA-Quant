"""Verify completed scope archive and extract original small logs safely."""
import argparse,hashlib,json,shutil,tarfile
from pathlib import Path
from datetime import datetime,timezone
p=argparse.ArgumentParser();p.add_argument('--backup',type=Path,required=True)
p.add_argument('--receipts',type=Path,required=True);p.add_argument('--raw-root',type=Path,required=True)
p.add_argument('--result-prefix',default='awq-scope-shard-')
p.add_argument('--expected-videos',type=int,required=True);a=p.parse_args();b=a.backup.resolve();checks=[]
for line in (b/'SHA256SUMS.txt').read_text().splitlines():
    digest,name=line.split(maxsplit=1);f=(b/name.strip()).resolve()
    assert f.parent==b and hashlib.sha256(f.read_bytes()).hexdigest()==digest,name
    checks.append(name)
videos=json.loads((b/'VIDEO_MANIFEST.json').read_text());assert len(videos)==a.expected_videos
receipt=dict(verified_at_utc=datetime.now(timezone.utc).isoformat(),backup_directory=str(b),
    files_verified=checks,video_count=len(videos),archive_bytes=(b/'session-results.tar.gz').stat().st_size,all_sha256_passed=True)
(b/'LOCAL_VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
a.receipts.mkdir(parents=True,exist_ok=True)
for name in ('SHA256SUMS.txt','RESULTS_SHA256SUMS.txt','VIDEO_MANIFEST.json','LARGE_FILES_NOT_IN_GIT.json','resume.json','LOCAL_VERIFICATION.json'):
    shutil.copyfile(b/name,a.receipts/name)
root=a.raw_root.resolve()
with tarfile.open(b/'session-results.tar.gz') as tf:
    for m in tf.getmembers():
        if not m.name.startswith('eval/'+a.result_prefix):continue
        f=(root/m.name[5:]).resolve();assert root in f.parents and (m.isdir() or m.isfile())
        if m.isdir():f.mkdir(parents=True,exist_ok=True)
        else:
            data=tf.extractfile(m).read()
            if f.exists():assert f.read_bytes()==data,str(f)
            else:f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(data)
print(json.dumps(receipt))
