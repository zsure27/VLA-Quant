"""Verify completed full download, safe extraction, both hashes and video/data coverage."""
import hashlib,json,os,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path.cwd();BASE=(ROOT/'backups/experiments/p5-contextual-routing/20261012-091-b5-pilot-v2').resolve()
NAME='091-gpu-b5-final-20261013-v1';packet=BASE/(NAME+'.tar.gz')
assert packet.stat().st_size==4201438552
h=hashlib.sha256()
with packet.open('rb') as f:
 for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
assert h.hexdigest()=='31ae496b9e14ba7573a703c7123cc8b690e5ffbde33c9b99fed965fdee8121ad'
target=BASE/NAME;assert not target.exists(), 'Do not overwrite an existing extraction'
longbase=Path('\\\\?\\'+str(BASE)) if os.name=='nt' and not str(BASE).startswith('\\\\?\\') else BASE
with tarfile.open(packet,'r:gz') as archive:archive.extractall(longbase,filter='data')
target=longbase/NAME
r=subprocess.run([sys.executable,'scripts/verify_vla_archive_local.py','--archive',str(target)],capture_output=True,text=True,check=True)
v=json.loads(r.stdout);assert v['files']=={'SHA256SUMS.txt':22244,'RESULTS_SHA256SUMS.txt':15389}
data=json.loads((target/'experiment/combined450-paired-analysis.json').read_text())
assert {(row['task_id'],row['init_state_index']) for row in data['rows']}=={(t,r) for t in range(10) for r in range(5,50)}
assert data['episodes']==450 and data['micro_excluded']
video_file=ROOT/'results/experiments/p5-contextual-routing/20261012-091-b5-pilot-v2/FINAL_VIDEO_COVERAGE.json'
video=json.loads(video_file.read_text());assert video['gate']=='PASS_FULL_ARCHIVE_ACTUAL_VIDEO_LOG_COVERAGE'
assert hashlib.sha256((target/'SHA256SUMS.txt').read_bytes()).hexdigest()==video['full_sha_manifest_sha256']
assert len(video['references'])==920
for row in video['references']:assert (target/row['archive']).is_file()
assert len(list((target/'baseline-video-increment').rglob('*.mp4')))==220
v.update(gate='PASS_LOCAL_FULL_SESSION_SHA_DATA_AND_VIDEO_COVERAGE',archive_alias=str(ROOT/'backups/experiments/p5-contextual-routing/20261012-091-b5-pilot-v2'/NAME),export_sha256=h.hexdigest(),export_bytes=packet.stat().st_size,formal_episodes_per_model=450,formal_total_with_random50=2750,original_video_count=1140,base_checkpoint_full_local_copy=False,server_archive='/root/autodl-tmp/qvla-repro/backups/'+NAME)
p=ROOT/'results/experiments/p5-contextual-routing/20261012-091-b5-pilot-v2/final-archive-local-verification.json'
with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2);f.write('\n')
print(json.dumps(v))
