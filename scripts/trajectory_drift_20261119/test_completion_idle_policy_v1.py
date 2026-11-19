import hashlib,tempfile,unittest
from pathlib import Path
from completion_idle_policy_v1 import completion_verified,idle_transition


class CompletionIdle(unittest.TestCase):
    def args(self,**change):
        args=dict(gpu_present=True,observation_valid=True,compute_processes_present=False,
                  owned_work_present=False,memory_used_mib=0,utilization_percent=0)
        args.update(change);return args
    def test_exact_continuous_thirty_minutes(self):
        since,close=idle_transition(None,100,**self.args());self.assertFalse(close)
        self.assertEqual(idle_transition(since,1899,**self.args()),(100,False))
        self.assertEqual(idle_transition(since,1900,**self.args()),(100,True))
    def test_activity_or_unknown_resets_timer(self):
        for change in [dict(gpu_present=False),dict(observation_valid=False),dict(compute_processes_present=True),dict(owned_work_present=True),dict(memory_used_mib=1000),dict(utilization_percent=1)]:
            with self.subTest(change=change):self.assertEqual(idle_transition(100,3000,**self.args(**change)),(None,False))
    def test_controller_receipt_must_match_task_and_recovery_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);index=root/'index.json';index.write_text('{}')
            auth=dict(task_manifest_sha256='fixture-task')
            proof=dict(task_manifest_sha256='fixture-task',round_id=root.name,
                gate='PASS_ALL_REQUESTED_TASK_WORK_AUDITED',all_requested_work_resolved=True,
                raw_audit_passed=True,recovery_index_relative_path='index.json',
                recovery_index_sha256=hashlib.sha256(index.read_bytes()).hexdigest())
            self.assertTrue(completion_verified(proof,auth,root))
            for change in [dict(round_id='old-round'),dict(raw_audit_passed=False),dict(all_requested_work_resolved=False),dict(task_manifest_sha256='other'),dict(recovery_index_relative_path='../outside')]:
                self.assertFalse(completion_verified(dict(proof,**change),auth,root))
            index.write_text('{"changed":true}')
            self.assertFalse(completion_verified(proof,auth,root))


if __name__=='__main__':unittest.main()
