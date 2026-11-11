"""Numerical identity and scale/reload contracts for the new unfused binding."""
import tempfile
from pathlib import Path
import unittest

import torch
from qvla.groot_sq_20261111_v1 import apply


class FakeQuant:
    @staticmethod
    def quantize_weight_per_channel_absmax(value,n_bits):
        q=2**(n_bits-1)-1
        scale=value.abs().amax(1,keepdim=True).clamp_min(1e-5)/q
        value.copy_((value/scale).round()*scale)

    @staticmethod
    def quantize_activation_per_token_absmax(value,n_bits):
        q=2**(n_bits-1)-1
        scale=value.abs().amax(-1,keepdim=True).clamp_min(1e-5)/q
        value.copy_((value/scale).round()*scale)


class Contract(unittest.TestCase):
    def model_and_profile(self):
        torch.manual_seed(7)
        model=torch.nn.Sequential(torch.nn.Linear(13,9))
        payload={'shapes':{'0':[9,13]},'scales':{'0':torch.logspace(-1,1,13)}}
        return model,payload

    def test_unquantized_identity_with_bias(self):
        model,profile=self.model_and_profile();x=torch.randn(2,5,13)
        expected=model(x)
        apply(model,profile,FakeQuant)
        torch.testing.assert_close(model(x),expected,rtol=2e-5,atol=2e-6)

    def test_precision_changes_and_profile_reload(self):
        model,profile=self.model_and_profile();x=torch.randn(2,5,13)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'profile.pt';torch.save(profile,path)
            restored=torch.load(path,weights_only=True)
        apply(model,restored,FakeQuant,4);actual=model(x)
        other,_=self.model_and_profile();apply(other,profile,FakeQuant,4)
        self.assertTrue(torch.equal(actual,other(x)))
        self.assertTrue(torch.isfinite(actual).all())
        eight,_=self.model_and_profile();apply(eight,profile,FakeQuant,8)
        self.assertFalse(torch.equal(actual,eight(x)))

    def test_wrong_weight_shape_rejected(self):
        model,profile=self.model_and_profile();profile['shapes']['0']=[10,13]
        with self.assertRaises(ValueError):apply(model,profile,FakeQuant)


if __name__=='__main__':unittest.main()
