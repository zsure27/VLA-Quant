from types import SimpleNamespace
import unittest
import numpy as np
from qvla.pi05_torch_backend_v2 import next_noise,noise_stream,policy_noise_shape


class NoiseContractTests(unittest.TestCase):
    def test_uses_declared_ten_step_block_and_query_pairing(self):
        policy=SimpleNamespace(_model=SimpleNamespace(config=SimpleNamespace(action_horizon=10,action_dim=32)))
        shape=policy_noise_shape(policy)
        a,b=noise_stream(),noise_stream()
        first=next_noise(a,shape)
        self.assertEqual(first.shape,(10,32))
        self.assertEqual(first.dtype,np.float32)
        self.assertTrue(np.array_equal(first,next_noise(b,shape)))
        second=next_noise(a,shape)
        self.assertFalse(np.array_equal(first,second))
        self.assertTrue(np.array_equal(second,next_noise(b,shape)))

    def test_invalid_dimensions_rejected_before_inference(self):
        for shape in [(0,32),(-1,32),(10,0),(10,),('10',32)]:
            with self.assertRaises(ValueError):next_noise(noise_stream(),shape)


if __name__=='__main__':unittest.main()
