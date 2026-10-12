"""Read recent primary rate metadata for one Codex task; never print messages/auth."""
from __future__ import annotations
import argparse, datetime, json, math, re
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--thread-id', required=True)
    p.add_argument('--max-age-seconds', type=int, default=300)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if not re.fullmatch(r'[0-9a-f-]{36}', a.thread_id) or not 1 <= a.max_age_seconds <= 300:
        raise ValueError('Require an exact task id and <=5 minute freshness')
    root = Path.home()/'.codex/sessions'
    files = [x for x in root.rglob('*'+a.thread_id+'*.jsonl') if x.is_file()]
    if not files: raise ValueError('Current task session unavailable')
    source = max(files, key=lambda x:x.stat().st_mtime)
    with source.open('rb') as stream:
        stream.seek(max(0,source.stat().st_size-3000000)); tail=stream.read()
    accepted=[]
    now=datetime.datetime.now(datetime.timezone.utc)
    for line in tail.splitlines():
        try: row=json.loads(line)
        except (ValueError,UnicodeDecodeError): continue
        body=row.get('payload',{})
        if row.get('type')!='event_msg' or not isinstance(body,dict) or body.get('type')!='token_count': continue
        limits=body.get('rate_limits') or {}; primary=limits.get('primary') or {}
        if limits.get('limit_id')!='codex' or primary.get('window_minutes')!=300: continue
        timestamp=datetime.datetime.fromisoformat(row['timestamp'].replace('Z','+00:00'))
        age=(now-timestamp).total_seconds(); used=primary.get('used_percent'); reset=primary.get('resets_at')
        if not (0<=age<=a.max_age_seconds and isinstance(used,(int,float)) and math.isfinite(used) and 0<=used<=100
                and isinstance(reset,(int,float)) and now.timestamp()<reset<=now.timestamp()+5*3600+300): continue
        accepted.append({'gate':'PASS_RECENT_CURRENT_TASK_PRIMARY_RESPONSE_METADATA',
                         'source':'Current task API response token_count.rate_limits; not an estimate or auth file',
                         'task_id':a.thread_id,'response_timestamp_utc':row['timestamp'],'read_timestamp_utc':now.isoformat(),
                         'age_seconds':age,'used_percent':used,'remaining_percent':100-used,'resets_at':reset,
                         'window_duration_minutes':300,'session_file':source.name})
    if not accepted: raise ValueError('No fresh authoritative task primary metadata; do not reuse history')
    value=accepted[-1]
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    print(json.dumps(value))


if __name__=='__main__': main()
