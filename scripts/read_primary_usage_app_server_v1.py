"""Read actual primary usage through the official local Codex app-server RPC.

No auth-file access, login, inference, reset credits, or raw account logging.
Only primary is inspected; other returned account fields are discarded.
"""
import argparse,datetime,json,queue,shutil,subprocess,threading,time

def read_primary(*,timeout=45,executable=None):
    started=time.monotonic();events=queue.Queue()
    proc=subprocess.Popen([executable or shutil.which('codex'),'app-server','--stdio'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,encoding='utf-8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    def reader():
        for line in proc.stdout:
            try:events.put(json.loads(line))
            except (ValueError,TypeError):continue
        events.put(dict(connection_closed=True))
    threading.Thread(target=reader,daemon=True).start()
    def request(number,method,params):
        proc.stdin.write(json.dumps(dict(id=number,method=method,params=params))+'\n');proc.stdin.flush()
        while True:
            left=timeout-(time.monotonic()-started)
            if left<=0:raise TimeoutError('Official primary RPC exceeded bounded timeout')
            message=events.get(timeout=left)
            if message.get('connection_closed'):raise RuntimeError('Official primary RPC connection closed')
            if message.get('id')==number:
                if 'error' in message:raise RuntimeError('Official primary RPC returned an error')
                return message['result']
    try:
        request(1,'initialize',dict(clientInfo=dict(name='vla_primary_readonly',version='1.0.0')))
        proc.stdin.write(json.dumps(dict(method='initialized'))+'\n');proc.stdin.flush()
        result=request(2,'account/rateLimits/read',dict(excludeResetCreditDetails=True,supportsLunaReserve=False))
        primary=(result.get('rateLimitsByLimitId',{}).get('codex') or result.get('rateLimits') or {}).get('primary')
        if not primary or primary.get('windowDurationMins')!=300:raise ValueError('Fresh 300-minute primary unavailable')
        used=primary['usedPercent'];reset=primary['resetsAt']
        if not 0<=used<=100 or not isinstance(reset,int) or reset<=time.time():raise ValueError('Primary window invalid or expired')
        return dict(source='codex_api',transport='official_app_server_account_rateLimits_read',window_minutes=300,observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),remaining_percent=100-used,resets_at=reset,rpc_latency_seconds=round(time.monotonic()-started,3))
    finally:
        proc.kill();proc.wait(timeout=10)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--timeout',type=int,default=45);a=p.parse_args()
    try:print(json.dumps(read_primary(timeout=a.timeout)))
    except Exception as e:
        print(json.dumps(dict(gate='FAIL_FRESH_PRIMARY_RPC',error_type=type(e).__name__,cached_quota_not_substituted=True)));raise SystemExit(1)
