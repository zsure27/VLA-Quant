import unittest
import numpy as np
from qvla.runtime_error_protocol_v3 import measure_pair,complete_case_ids,reflected_kde


class RuntimeErrorV3(unittest.TestCase):
    def setup_pair(self,model='pi05'):
        r=dict(model=model,precision='SQ-W4A8',suite='libero_goal',task_id=0,reset_index=5,
            checkpoint_sha256='c',quantizer_sha256='q',normalization_sha256='n',model_seed=7,env_seed=0,
            matched_noise_sha256='noise',execute_horizon=5 if model=='pi05' else 1,
            query_timebase='native_policy_query',coordinate_frame='world',perturbed=False)
        p=dict(r,perturbed=True)
        scales=dict(action_channels=[1]*6,translation_m=.01,rotation_rad=.1)
        return r,p,scales
    def call(self,r,p,s,next_query=3):
        a=np.zeros((8,7));b=a.copy();b[:,0]=2;b[:,6]=999
        b[r['execute_horizon']:,:6]=999  # Unexecuted suffix must not affect the metric.
        return measure_pair(r,p,[0,0,0],[.01,0,0],[0,0,0,1],[0,0,0,1],a,b,s,2,next_query)
    def test_same_precision_tplusone_and_gripper_exclusion(self):
        r,p,s=self.setup_pair();x=self.call(r,p,s)
        self.assertEqual(x['epsilon_T'],1);self.assertEqual(x['delta_A_all_next'],2)
        self.assertEqual(x['sensitivity_T'],2);self.assertIsNone(x['sensitivity_R'])
    def test_cross_precision_reference_rejected(self):
        r,p,s=self.setup_pair();r['precision']='BF16'
        with self.assertRaises(ValueError):self.call(r,p,s)
    def test_nonfinite_scales_rejected(self):
        r,p,s=self.setup_pair();s['translation_m']=float('nan')
        with self.assertRaises(ValueError):self.call(r,p,s)
    def test_current_query_instead_of_next_rejected(self):
        r,p,s=self.setup_pair()
        with self.assertRaises(ValueError):self.call(r,p,s,next_query=2)
    def test_native_horizon_and_noise_mismatch_rejected(self):
        r,p,s=self.setup_pair();r['execute_horizon']=p['execute_horizon']=8
        with self.assertRaises(ValueError):self.call(r,p,s)
        r,p,s=self.setup_pair('groot_n15');p['matched_noise_sha256']='wrong'
        with self.assertRaises(ValueError):self.call(r,p,s)
    def test_complete_case_and_missing_counts(self):
        rows=[dict(case_id=c,precision=p,tau=t) for c in ('a','b') for p in ('FP','Q') for t in (0,2)]
        rows=[r for r in rows if (r['case_id'],r['precision'],r['tau'])!=('b','Q',2)]
        complete,risk=complete_case_ids(rows,['FP','Q'],[0,2])
        self.assertEqual(complete,{'a'});self.assertEqual(risk['Q']['2'],1)
        with self.assertRaises(ValueError):complete_case_ids(rows+[rows[0]],['FP','Q'],[0,2])
    def test_reflected_density_nonnegative_and_normalized(self):
        grid=np.linspace(0,5,180);z=reflected_kde([[.1,.2],[1,1]],grid,grid,[.2,.3])
        self.assertTrue(np.all(z>=0))
        integral=np.trapezoid(np.trapezoid(z,grid,axis=0),grid)
        self.assertAlmostEqual(float(integral),1,places=2)


if __name__=='__main__':unittest.main()
