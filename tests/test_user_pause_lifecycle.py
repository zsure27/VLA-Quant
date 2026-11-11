import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from qvla.infra.lifecycle import finish_round


class PauseLifecycleTests(unittest.TestCase):
    def test_pause_never_calls_native_shutdown(self):
        with tempfile.TemporaryDirectory() as temp, patch('qvla.infra.lifecycle.native_shutdown') as shutdown:
            result=finish_round(Path(temp),{'pause_only':True,'shutdown_authorized':False},
                                {'shutdown_provider':'user_pause'})
            self.assertEqual(result,'PAUSED_SERVER_LEFT_ON')
            self.assertTrue((Path(temp)/'pause-receipt.json').is_file())
            shutdown.assert_not_called()

    def test_revoked_shutdown_cannot_use_native_provider(self):
        with tempfile.TemporaryDirectory() as temp, patch('qvla.infra.lifecycle.native_shutdown') as shutdown:
            with self.assertRaises(ValueError):
                finish_round(Path(temp),{'pause_only':True,'shutdown_authorized':False},
                             {'shutdown_provider':'autodl_native'})
            shutdown.assert_not_called()


if __name__=='__main__':unittest.main()
