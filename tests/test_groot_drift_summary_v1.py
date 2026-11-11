import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('_groot_summary',Path(__file__).resolve().parents[1]/'scripts/trajectory_drift_20261111/summarize_groot_pilots_v1.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Contract(unittest.TestCase):
    def test_physical_units_common_prefix_and_quaternion_sign(self):
        left=({},np.zeros((3,3)),np.array([[0,0,0,1]]*3),None)
        right=({},np.array([[.003,.004,0]]*2),np.array([[0,0,0,-1]]*2),None)
        result=module.drift(left,right)
        self.assertEqual(result['observed_common_prefix_steps'],2)
        self.assertEqual(result['physical_control_step'],[1,2])
        np.testing.assert_allclose(result['translation_error_mm'],[5,5])
        np.testing.assert_allclose(result['rotation_geodesic_deg'],[0,0])

    def test_persistence_not_single_spike(self):
        self.assertIsNone(module.first_window([False,True,False,True,True,True,True]))
        self.assertEqual(module.first_window([False,True,True,True,True,True]),2)


if __name__=='__main__':unittest.main()
