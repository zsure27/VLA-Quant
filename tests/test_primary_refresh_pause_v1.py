from datetime import datetime,timedelta,timezone
import signal
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock

from qvla.infra.lifecycle import quota_decision
from qvla.infra.runner import confirmed_quota


class RefreshContract(unittest.TestCase):
    def setUp(self):
        self.now=datetime.now(timezone.utc)
        self.control=dict(close_remaining_percent=10,stop_expansion_remaining_percent=15,
                          quota_max_age_seconds=1020,refresh_confirmation_wait_seconds=90)
        self.old=dict(source='codex_api',window_minutes=300,observed_at=(self.now-timedelta(seconds=100)).isoformat(),
                      remaining_percent=33,resets_at=self.now.timestamp()-2)

    def test_expired_window_never_means_new_quota(self):
        self.assertEqual(quota_decision(self.old,self.control,self.now),'WAIT_API_REFRESH')
        self.old['resets_at']=self.now.timestamp()-91
        self.assertEqual(quota_decision(self.old,self.control,self.now),'CLOSE')

    def test_new_actual_api_resumes_and_ten_percent_still_closes(self):
        self.old.update(resets_at=self.now.timestamp()+18000,remaining_percent=100)
        self.assertEqual(quota_decision(self.old,self.control,self.now),'CONTINUE')
        self.old.update(resets_at=self.now.timestamp()-1,remaining_percent=10)
        self.assertEqual(quota_decision(self.old,self.control,self.now),'CLOSE')

    def test_real_api_required_while_process_group_paused(self):
        child=SimpleNamespace(pid=123,poll=lambda:None);publish=Mock()
        with patch('qvla.infra.runner.os.killpg',create=True) as kill,patch('qvla.infra.runner.time.sleep'), \
             patch.object(signal,'SIGSTOP',19,create=True),patch.object(signal,'SIGCONT',18,create=True):
            result=confirmed_quota('WAIT_API_REFRESH',lambda:'CONTINUE',self.control,publish,[],child)
        self.assertEqual(result,'CONTINUE')
        self.assertEqual(kill.call_args_list[0].args,(123,19))
        self.assertEqual(kill.call_args_list[-1].args,(123,18))
        self.assertEqual(publish.call_args_list[0].kwargs['state'],'WAITING_PRIMARY_API')

    def test_stale_api_or_stop_does_not_extend_work(self):
        self.old['observed_at']=(self.now-timedelta(seconds=1021)).isoformat()
        with self.assertRaises(ValueError):quota_decision(self.old,self.control,self.now)
        child=SimpleNamespace(pid=123,poll=lambda:None)
        with patch('qvla.infra.runner.os.killpg',create=True),patch('qvla.infra.runner.time.sleep'), \
             patch.object(signal,'SIGSTOP',19,create=True),patch.object(signal,'SIGCONT',18,create=True):
            self.assertEqual(confirmed_quota('WAIT_API_REFRESH',lambda:'CONTINUE',self.control,Mock(),[15],child),'CLOSE')

    def test_missing_confirmation_has_bounded_deadline(self):
        child=SimpleNamespace(pid=123,poll=lambda:None);read=Mock(return_value='WAIT_API_REFRESH')
        with patch('qvla.infra.runner.os.killpg',create=True), \
             patch.object(signal,'SIGSTOP',19,create=True),patch.object(signal,'SIGCONT',18,create=True), \
             patch('qvla.infra.runner.time.monotonic',side_effect=[0,91]):
            self.assertEqual(confirmed_quota('WAIT_API_REFRESH',read,self.control,Mock(),[],child),'CLOSE')
        read.assert_not_called()


if __name__=='__main__':unittest.main()
