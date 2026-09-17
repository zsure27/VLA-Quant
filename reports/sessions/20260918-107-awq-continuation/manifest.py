"""Write or verify this session's source-data and figure manifest."""
import argparse,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args()
files={}
for base in (HERE,ROOT/'results/107-awq-continuation-20260918'):
    for f in sorted(base.rglob('*')):
        if not f.is_file() or f.name=='manifest.json' or '__pycache__' in f.parts:continue
        b=f.read_bytes()
        if f.suffix in ('.md','.py','.json','.jsonl','.csv','.txt','.log','.svg'):b=b.replace(b'\r\n',b'\n')
        files[f.relative_to(ROOT).as_posix()]=hashlib.sha256(b).hexdigest()
target=HERE/'manifest.json'
if a.verify:
    assert json.loads(target.read_text())['files']==files,'session manifest mismatch'
else:target.write_text(json.dumps(dict(normalization='LF for text; bytes otherwise',files=files),indent=2)+'\n')
print('Verified' if a.verify else 'Wrote',len(files),'file hashes')
