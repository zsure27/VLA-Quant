"""Bind final experiment videos to actual completed evaluation log references."""
import hashlib,json
from pathlib import Path
ROOT=Path('/root/autodl-tmp/qvla-repro');BASE=ROOT/'backups/experiments/p5-contextual-routing/20261012-091-b5-pilot-v2';FULL=ROOT/'backups/091-gpu-b5-final-20261013-v1'
manifest={}
for line in (FULL/'SHA256SUMS.txt').read_text().splitlines():
 h,path=line.split(maxsplit=1);manifest[path]=h
refs=[];counts={}
for phase,n in [('micro',10),('first50',50),('reset-10-19',100),('reset-20-29',100),('reset-30-39',100),('reset-40-49',100)]:
 for policy in ('BF16','A0'):
  logs=list((BASE/phase/policy).glob('EVAL-*.txt'));assert len(logs)==1
  rows=[Path(line.split('Saved rollout MP4 at path ',1)[1].strip()) for line in logs[0].read_text().splitlines() if 'Saved rollout MP4 at path ' in line]
  assert len(rows)==n
  counts[phase+'/'+policy]=len(rows)
  for row in rows:
   assert not row.is_absolute() and '..' not in row.parts
   rel=(Path('frozen-control-v2/source')/row).as_posix();f=FULL/rel
   assert f.is_file() and rel in manifest
   refs.append({'phase':phase,'policy':policy,'archive':rel,'sha256':manifest[rel]})
assert len(refs)==len({r['archive'] for r in refs})==920
p0=list((FULL/'baseline-video-increment').rglob('*.mp4'));assert len(p0)==220
assert all(p.relative_to(FULL).as_posix() in manifest for p in p0)
v={'gate':'PASS_FULL_ARCHIVE_ACTUAL_VIDEO_LOG_COVERAGE','P5_BF16_A0_video_count':920,'formal_P5_videos':900,'micro_videos':20,'baseline_bridge_videos':220,'unique_original_video_total':1140,'phase_counts':counts,'references':refs,'full_sha_manifest_sha256':hashlib.sha256((FULL/'SHA256SUMS.txt').read_bytes()).hexdigest(),'limitation':'B5/B3/B4/task-only evaluator did not create videos; full raw observations/actions/traces are retained'}
with (BASE/'FINAL_VIDEO_COVERAGE.json').open('x') as f:json.dump(v,f,indent=2);f.write('\n')
print(json.dumps({k:v[k] for k in v if k!='references'}))
