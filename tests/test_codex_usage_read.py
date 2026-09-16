import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from codex_usage_read import normalize, decision, read_session


class QuotaContractTest(unittest.TestCase):
    def payload(self, used=85):
        return {"accountId": "test-only-account", "rateLimits": {"primary": {"usedPercent": 1}},
            "rateLimitsByLimitId": {"codex": {"limitId": "codex", "primary": {
                "usedPercent": used, "windowDurationMins": 300, "resetsAt": 123},
                "secondary": {"usedPercent": 22, "windowDurationMins": 10080, "resetsAt": 456}}}}

    def test_primary_multi_bucket_used_is_not_remaining(self):
        value = normalize(self.payload(), "test")
        self.assertEqual(value["windows"]["primary"]["remaining_percent"], 15)
        self.assertNotIn("accountId", value)
        self.assertEqual(decision(value)["action"], "LIVE_BUDGET_AVAILABLE")

    def test_stale_and_missing_are_unknown(self):
        value = normalize(self.payload(), "test")
        now = datetime.fromisoformat(value["observed_at_utc"])
        self.assertEqual(decision(value, now=now+timedelta(seconds=301))["action"], "UNKNOWN_DO_NOT_DISPATCH")
        value["windows"]["secondary"] = None
        self.assertEqual(decision(value)["action"], "UNKNOWN_DO_NOT_DISPATCH")

    def test_threshold_and_next_test_preserve_shutdown_reserve(self):
        value = normalize(self.payload(91), "test")
        self.assertEqual(decision(value)["action"], "BACKUP_AND_CLOSE")
        value = normalize(self.payload(85), "test")
        self.assertEqual(decision(value, estimated_next=13)["action"], "BACKUP_AND_CLOSE")

    def test_invalid_usage_is_not_accepted(self):
        with self.assertRaises(ValueError): normalize(self.payload(float("nan")), "test")
        with self.assertRaises(ValueError): normalize(self.payload(True), "test")

    def test_response_event_preserves_age_and_excludes_other_session(self):
        thread = "00000000-0000-4000-8000-000000000001"
        timestamp = (datetime.now(timezone.utc)-timedelta(seconds=30)).isoformat()
        event = {"type": "event_msg", "timestamp": timestamp, "payload": {
            "type": "token_count", "rate_limits": {"limit_id": "codex",
             "primary": {"used_percent": 85, "window_minutes": 300},
             "secondary": {"used_percent": 22, "window_minutes": 10080}}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/("rollout-"+thread+".jsonl")
            path.write_text(json.dumps({"type": "session_meta", "payload": {"id": thread}})+"\n"+json.dumps(event)+"\n")
            value = read_session(root, thread)
            self.assertEqual(value["observed_at_utc"], timestamp)
            self.assertEqual(value["windows"]["primary"]["remaining_percent"], 15)
            self.assertGreaterEqual(decision(value)["age_seconds"], 30)
            segment = root/("rollout-"+thread+"_core-segment.jsonl")
            segment.write_text(path.read_text())
            path.unlink()
            self.assertEqual(read_session(root, thread)["observed_at_utc"], timestamp)
            path = segment
            path.write_text(json.dumps({"type": "session_meta", "payload": {"id": "other"}})+"\n"+json.dumps(event)+"\n")
            with self.assertRaises(ValueError): read_session(root, thread)


if __name__ == "__main__": unittest.main()
