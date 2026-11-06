import unittest
import numpy as np
from qvla.runtime_error_protocol_v4 import HORIZONS,measure_pair
from qvla.runtime_error_protocol_v3 import measure_pair as measure_v3


class OftRuntimeErrorV4(unittest.TestCase):
    def inputs(self,model):
        r=dict(model=model,precision='SQ-W4A8',suite='libero_spatial',task_id=0,reset_index=5,
               checkpoint_sha256='checkpoint',quantizer_sha256='profile',normalization_sha256='scale',
               model_seed=7,env_seed=0,matched_noise_sha256='noise',execute_horizon=HORIZONS[model],
               query_timebase='native_policy_query',coordinate_frame='world',perturbed=False)
        p=dict(r,perturbed=True)
        a=np.zeros((9,7));b=a.copy();b[7,0]=8;b[8,0]=1000;b[:,6]=1000
        return [r,p,[0,0,0],[.1,0,0],[0,0,0,1],[0,0,0,1],a,b,
                dict(action_channels=[1]*6,translation_m=1,rotation_rad=1),2,3]

    def test_oft_uses_eighth_executed_slot_and_excludes_tail_and_gripper(self):
        result=measure_pair(*self.inputs('openvla_oft'))
        self.assertAlmostEqual(result['delta_A_all_next'],np.sqrt(8))
        self.assertIsNone(result['sensitivity_R'])
        for model in ('pi05','groot_n15'):
            self.assertEqual(measure_pair(*self.inputs(model))['delta_A_all_next'],0)

    def test_model_and_execution_schedule_must_match(self):
        values=self.inputs('openvla_oft');values[1]['model']='pi05'
        with self.assertRaises(ValueError):measure_pair(*values)
        values=self.inputs('openvla_oft');values[0]['execute_horizon']=values[1]['execute_horizon']=5
        with self.assertRaises(ValueError):measure_pair(*values)

    def test_frozen_v3_remains_two_model_protocol(self):
        with self.assertRaises(ValueError):measure_v3(*self.inputs('openvla_oft'))


if __name__=='__main__':unittest.main()
