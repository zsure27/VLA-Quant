import unittest
import numpy as np
from qvla.trajectory_diagnostics_v2 import (factorial_errors,boundary_projection,event_aligned,
    activation_summary,demo_symmetric_qdq,first_close_command)
from qvla.run_trajectory_diagnostic_v2 import validate_admission


class DiagnosticsV2(unittest.TestCase):
    def test_factorial_reconstruction_and_interaction(self):
        f=np.zeros((2,8,7));w=f+.1;a=f+.2;wa=f+.45
        x=factorial_errors(f,w,a,wa,["h"]*4,["c"]*4)
        np.testing.assert_allclose(x["interaction"],.15)
        np.testing.assert_allclose(x["weight"]+x["activation"]+x["interaction"],x["total"])

    def test_factorial_rejects_mixed_coordinate_and_input(self):
        f=np.zeros((2,8,7))
        with self.assertRaises(ValueError):factorial_errors(f,f,f,f,["a","b","a","a"],["c"]*4)
        with self.assertRaises(ValueError):factorial_errors(f,f,f,f,["a"]*4,["c","d","c","c"])

    def test_projection_keeps_sign_excludes_gripper(self):
        t=np.zeros((1,3,7));t[0,:,0]=[0,1,2]
        s=t.copy();s[...,0]-=.5;s[...,6]=9999
        x=boundary_projection(s,t,np.ones(6))
        np.testing.assert_allclose(x["signed_parallel"],-.5)
        np.testing.assert_allclose(x["perpendicular"],0)
        np.testing.assert_allclose(x["angle_deg"],180)

    def test_zero_change_masked_instead_of_false_angle(self):
        t=np.zeros((1,3,7));x=boundary_projection(t,t,np.ones(6))
        self.assertFalse(x["valid"].any());self.assertTrue(np.isnan(x["angle_deg"]).all())

    def test_event_missing_and_horizon_counts(self):
        x=event_aligned([[1,2,3],[4,5],[8]],[1,0,None],before=1,after=1)
        self.assertEqual(x["counts"].tolist(),[1,2,2]);self.assertEqual(x["no_event"],1)
        c=np.zeros((5,7));c[:,-1]=[-1,-1,1,1,-1]
        self.assertEqual(first_close_command(c),2)

    def test_qdq_and_out_of_range_are_distinct_from_precision(self):
        a=np.array([-9.,-.3,0.,.3,9.]);q=demo_symmetric_qdq(a,4,1.)
        self.assertTrue(np.max(np.abs(q))<=1.)
        s=activation_summary(a,1.,q);self.assertEqual(s["out_of_calibration_range_fraction"],.4)
        self.assertGreater(s["paired_mse"],0)

    def test_activation_backend_requires_smoothing_gate(self):
        identity=dict(data_kind="measured",backend="smoothquant",checkpoint="model",weight_bits=4,activation_bits=8,profile_sha256={"profile":"sha"},coordinate_id="coordinate")
        gate=dict(gate="PASS_TRAJECTORY_CAPTURE_ACTUAL_MODEL_V2",identity=identity,
            **{k:True for k in ("same_actions","same_observations","same_seeds","same_query_schedule","finite","frozen","reload")})
        argv=["--pretrained_checkpoint","model","--seed-protocol","paired","--method","smoothquant","--weight-bits","4","--activation-bits","8","--profile","profile"]
        with self.assertRaises(ValueError):validate_admission(identity,gate,argv)
        gate["smooth_only_equivalence"]=dict(gate="PASS_SQ_SMOOTH_ONLY_EQUIVALENCE_V2",coordinate_id="coordinate",passed=True)
        self.assertEqual(validate_admission(identity,gate,argv),"smoothquant")


if __name__=="__main__":unittest.main()
