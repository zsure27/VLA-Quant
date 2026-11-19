"""Offline safety regression tests; no account RPC, SSH, GPU or lease renewal."""
import contextlib
import datetime
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import primary_usage_relay_v5 as relay


class RelayTests(unittest.TestCase):
    def setUp(self):
        self.now = relay.utc_now()
        self.expires = (self.now+datetime.timedelta(hours=3)).isoformat()
        self.binding = dict(quota_only=True, hostname='fixture-host', boot_id='fixture-boot',
            gpu_uuid='fixture-gpu', authorization_sha256='fixture-sha',
            authorization_expires_at=self.expires,
            authorization_deadline_source='Fixture explicit planned operating window')
        self.snapshot = dict(hostname='fixture-host', boot_id='fixture-boot',
            gpu_uuid='fixture-gpu', authorization_sha256='fixture-sha',
            authorization_expires_at=self.expires, guard_matches=True,
            guard_state='CONTINUATION_ARMED')
        self.quota = dict(source='codex_api', window_minutes=300,
            observed_at=self.now.isoformat(), remaining_percent=80,
            resets_at=int(self.now.timestamp())+900)

    def test_preflight_rejects_run_overrunning_lease(self):
        with self.assertRaisesRegex(relay.RelayRejected, 'LEASE_INSUFFICIENT'):
            relay.preflight(self.binding, 10800, 900, self.now)

    def test_preflight_requires_explicit_deadline(self):
        del self.binding['authorization_expires_at']
        with self.assertRaises(KeyError):
            relay.preflight(self.binding, 600, 900, self.now)

    def test_preflight_is_not_remote_pass(self):
        self.assertFalse(relay.preflight(self.binding, 600, 900, self.now)['remote_delivery_verified'])

    def test_expired_lease_classified_terminal(self):
        with self.assertRaisesRegex(relay.RelayRejected, 'AUTHORIZATION_EXPIRED') as error:
            relay.validate_snapshot(self.binding, self.snapshot, self.quota,
                                    self.now+datetime.timedelta(hours=4))
        self.assertFalse(error.exception.retryable)

    def test_binding_mismatches_and_closing_are_terminal(self):
        for field, value in [('hostname','other'), ('boot_id','other'), ('gpu_uuid','other'),
                             ('authorization_sha256','other'), ('guard_matches',False),
                             ('guard_state','CLOSING'), ('authorization_expires_at',self.now.isoformat())]:
            with self.subTest(field=field):
                with self.assertRaises(relay.RelayRejected) as error:
                    relay.validate_snapshot(self.binding, dict(self.snapshot, **{field:value}), self.quota, self.now)
                self.assertFalse(error.exception.retryable)

    def test_stale_quota_retryable_without_timestamp_rewrite(self):
        old = dict(self.quota, observed_at=(self.now-datetime.timedelta(seconds=90)).isoformat())
        with self.assertRaisesRegex(relay.RelayRejected, 'QUOTA_STALE') as error:
            relay.validate_snapshot(self.binding,self.snapshot,old,self.now)
        self.assertTrue(error.exception.retryable)
        self.assertNotEqual(old['observed_at'],self.now.isoformat())

    def test_weekly_or_invalid_quota_rejected(self):
        for changed in [dict(weekly={}),dict(remaining_percent=101),dict(resets_at=1),dict(window_minutes=60)]:
            with self.subTest(changed=changed), self.assertRaises(relay.RelayRejected):
                relay.validate_snapshot(self.binding,self.snapshot,dict(self.quota,**changed),self.now)

    def test_remote_cannot_mark_identity_failure_retryable(self):
        with self.assertRaises(relay.RelayRejected) as error:
            relay.decode_reply(json.dumps(dict(ok=False,code='GPU_CHANGED',retryable=True)))
        self.assertFalse(error.exception.retryable)

    def test_malformed_reply_is_transient_redacted(self):
        with self.assertRaisesRegex(relay.RelayRejected,'REMOTE_REPLY_UNAVAILABLE') as error:
            relay.decode_reply('fixture private stderr, not a JSON response')
        self.assertTrue(error.exception.retryable)
        self.assertNotIn('private',str(error.exception))

    def execute_remote(self, *, expired=False, closing=False):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'stages'/'one').mkdir(parents=True)
            (root/'proc').mkdir()
            auth=dict(expires_at=(self.now-datetime.timedelta(seconds=5)).isoformat() if expired else self.expires)
            auth_bytes=json.dumps(auth).encode();(root/'authorization.json').write_bytes(auth_bytes)
            (root/'guard-status.json').write_text(json.dumps(dict(state='CLOSING' if closing else 'CONTINUATION_ARMED')))
            (root/'proc'/'boot').write_text('fixture-boot')
            (root/'proc'/'cmdline').write_bytes(str(root).encode())
            (root/'quota.json').write_text('ORIGINAL')
            stage=root/'stages'/'one'/'quota.json';stage.write_text('ORIGINAL')
            binding=dict(self.binding,control=str(root),guard_pid=123,
                authorization_sha256=hashlib.sha256(auth_bytes).hexdigest(),
                authorization_expires_at=auth['expires_at'])
            program=relay.remote_program(binding,self.quota)
            # Use the concrete platform class: Python 3.11 Path.__new__ reads
            # the module-level Path symbol, which this fixture patches below.
            real_path=type(root)
            def mapped(*parts):
                path=real_path(*parts)
                if str(path).replace('\\','/')=='/proc/sys/kernel/random/boot_id':
                    return root/'proc'/'boot'
                if str(path).replace('\\','/')=='/proc/123/cmdline':
                    return root/'proc'/'cmdline'
                return path
            out=io.StringIO()
            with patch('pathlib.Path',side_effect=mapped), patch('socket.gethostname',return_value='fixture-host'), \
                 patch('subprocess.check_output',return_value='fixture-gpu\n'), contextlib.redirect_stdout(out):
                exec(compile(program,'remote-fixture','exec'),{})
            return json.loads(out.getvalue()),(root/'quota.json').read_text(),stage.read_text(),(root/'authorization.json').read_bytes()==auth_bytes

    def test_actual_generated_remote_program_delivers_both_quotas(self):
        result,root,stage,unchanged=self.execute_remote()
        self.assertTrue(result['ok']);self.assertEqual(json.loads(root),self.quota)
        self.assertEqual(root,stage);self.assertTrue(unchanged)

    def test_actual_expired_remote_program_writes_no_quota(self):
        result,root,stage,unchanged=self.execute_remote(expired=True)
        self.assertEqual(result['code'],'AUTHORIZATION_EXPIRED')
        self.assertEqual(root,'ORIGINAL');self.assertEqual(stage,'ORIGINAL');self.assertTrue(unchanged)

    def test_actual_closing_remote_program_writes_no_quota(self):
        result,root,stage,unchanged=self.execute_remote(closing=True)
        self.assertEqual(result['code'],'CONTROL_CLOSING_NO_WRITE')
        self.assertEqual(root,'ORIGINAL');self.assertEqual(stage,'ORIGINAL');self.assertTrue(unchanged)


if __name__=='__main__':
    unittest.main()
