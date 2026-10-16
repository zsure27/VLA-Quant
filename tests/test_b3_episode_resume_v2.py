import types
import unittest
from qvla.b3_episode_resume_v2 import MISSING, selected_run_task, validate_partition


class ResumeScheduleTest(unittest.TestCase):
    def test_only_selected_task_preserves_original_cfg_and_episode_seed_inputs(self):
        cfg = types.SimpleNamespace(initial_state_offset=7, num_trials_per_task=3,
                                    seed_protocol='paired', seed=0, env_seed=1)
        calls, visited = [], []
        def task(cfg, task_id, total_episodes, total_successes):
            calls.append((cfg, task_id, cfg.seed, cfg.env_seed))
            return total_episodes + 3, total_successes
        wrapped = selected_run_task(task, visited)
        totals = (0, 0)
        for t in range(10):
            totals = wrapped(cfg, t, *totals)
        self.assertEqual(totals, (3, 0))
        self.assertEqual(visited, list(MISSING))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1:], (9, 0, 1))
        self.assertIsNot(calls[0][0], cfg)
        self.assertEqual(vars(calls[0][0]), vars(cfg))
        with self.assertRaises(ValueError):
            wrapped(cfg, 9, *totals)

    def test_partition_refuses_overlap_gap_or_repeated_episode(self):
        completed = [(t, r) for t in range(10) for r in range(5, 10)][:47]
        validate_partition(completed, MISSING)
        for bad in (completed[:-1], completed + [(9, 7)], completed[::-1]):
            with self.assertRaises(ValueError):
                validate_partition(bad, MISSING)
        with self.assertRaises(ValueError):
            validate_partition(completed, ((9, 6), (9, 8), (9, 9)))

    def test_upstream_or_changed_count_is_refused_before_running(self):
        wrapped = selected_run_task(lambda cfg, task_id, total_episodes, total_successes: None, [])
        with self.assertRaises(ValueError):
            wrapped(types.SimpleNamespace(initial_state_offset=7, num_trials_per_task=3,
                                          seed_protocol='upstream'), 9, 0, 0)


if __name__ == '__main__':
    unittest.main()
