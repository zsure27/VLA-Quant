import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('_round_event',Path(__file__).resolve().parents[1]/'scripts/trajectory_drift_20261111/wait_round_transition_v2.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class EventContract(unittest.TestCase):
    def test_already_failed_returns_without_waiting(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'status.json').write_text(json.dumps(dict(state='INTERRUPTED',current_stage='object')))
            with patch.object(module.ctypes,'CDLL',side_effect=AssertionError('Must not start event wait')):
                self.assertEqual(module.wait_transition(root)['event'],'ALREADY_TERMINAL')

    def test_same_running_stage_is_not_transition(self):
        state=dict(state='RUNNING',current_stage='spatial')
        self.assertFalse(module.changed(state,state))
        self.assertTrue(module.changed(dict(state='RUNNING',current_stage='object'),state))
        self.assertTrue(module.changed(dict(state='SHUTDOWN_PENDING',current_stage='spatial'),state))


if __name__=='__main__':unittest.main()
