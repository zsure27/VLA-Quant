import datetime,unittest
from primary_only_closure_policy_v1 import decision


class QuotaOnlyPolicy(unittest.TestCase):
    def setUp(self):
        self.now=datetime.datetime.now(datetime.timezone.utc)
        self.q=dict(source='codex_api',window_minutes=300,observed_at=self.now.isoformat(),remaining_percent=80,resets_at=int(self.now.timestamp())+600)
    def test_only_actual_threshold_closes(self):
        for value,expected in [(100,'CONTINUE'),(16,'CONTINUE'),(15,'STOP_EXPANSION_SAVE'),(11,'STOP_EXPANSION_SAVE'),(10,'CLOSE_FRESH_PRIMARY_THRESHOLD'),(3,'CLOSE_FRESH_PRIMARY_THRESHOLD'),(0,'CLOSE_FRESH_PRIMARY_THRESHOLD')]:
            with self.subTest(value=value):self.assertEqual(decision(dict(self.q,remaining_percent=value),now=self.now),expected)
    def test_stale_missing_and_weekly_cannot_close(self):
        for q in [None,{},dict(self.q,window_minutes=10080,remaining_percent=0),dict(self.q,observed_at=(self.now-datetime.timedelta(seconds=1021)).isoformat(),remaining_percent=0),dict(self.q,resets_at=1,remaining_percent=0)]:
            with self.subTest(q=q):self.assertEqual(decision(q,now=self.now),'PAUSE_SAVE_ALERT_PRIMARY_UNAVAILABLE')
    def test_authorization_expiry_pauses_not_closes(self):
        self.assertEqual(decision(self.q,now=self.now,authorization_valid=False),'PAUSE_SAVE_ALERT_AUTHORIZATION')
    def test_invalid_numbers_cannot_trigger_shutdown(self):
        for value in [-1,101,float('nan'),float('inf'),True,'0']:
            with self.subTest(value=value):self.assertEqual(decision(dict(self.q,remaining_percent=value),now=self.now),'PAUSE_SAVE_ALERT_PRIMARY_UNAVAILABLE')


if __name__=='__main__':unittest.main()
