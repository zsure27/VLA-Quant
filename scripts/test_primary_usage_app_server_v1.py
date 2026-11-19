"""Failure/field isolation tests; no real account/network calls."""
import io,json,time,unittest
from unittest.mock import patch
import read_primary_usage_app_server_v1 as primary

class Process:
    def __init__(self,result=None,error=None):
        self.stdin=io.StringIO();self.killed=False
        response=dict(id=2,result=result) if error is None else dict(id=2,error=error)
        self.stdout=io.StringIO(json.dumps(dict(id=1,result={}))+'\n'+json.dumps(response)+'\n')
    def kill(self):self.killed=True
    def wait(self,**kwargs):return 0

class PrimaryContract(unittest.TestCase):
    def read(self,body,error=None):
        child=Process(body,error)
        try:
            with patch.object(primary.subprocess,'Popen',return_value=child):
                return primary.read_primary(timeout=1,executable='fixture-only-codex')
        finally:
            self.assertTrue(child.killed)
            messages=[json.loads(x) for x in child.stdin.getvalue().splitlines()]
            request=next(x for x in messages if x.get('method')=='account/rateLimits/read')
            self.assertEqual(request['params'],dict(excludeResetCreditDetails=True,supportsLunaReserve=False))
    def body(self,**kwargs):
        p=dict(usedPercent=20,windowDurationMins=300,resetsAt=int(time.time())+600);p.update(kwargs)
        return dict(rateLimitsByLimitId=dict(codex=dict(primary=p,secondary={'excluded_fixture':True})))
    def test_valid_filters_other_fields(self):
        result=self.read(self.body());self.assertEqual(result['remaining_percent'],80)
        self.assertNotIn('secondary',result);self.assertNotIn('weekly',result);self.assertNotIn('credits',result)
    def test_wrong_window(self):
        with self.assertRaises(ValueError):self.read(self.body(windowDurationMins=60))
    def test_expired_window(self):
        with self.assertRaises(ValueError):self.read(self.body(resetsAt=int(time.time())-1))
    def test_missing_primary(self):
        with self.assertRaises(ValueError):self.read(dict(rateLimitsByLimitId=dict(codex={})))
    def test_impossible_used(self):
        with self.assertRaises(ValueError):self.read(self.body(usedPercent=101))
    def test_rpc_error_redacted(self):
        with self.assertRaisesRegex(RuntimeError,'Official primary RPC returned an error') as found:self.read(None,error=dict(message='private fixture content'))
        self.assertNotIn('private fixture content',str(found.exception))

if __name__=='__main__':unittest.main()
