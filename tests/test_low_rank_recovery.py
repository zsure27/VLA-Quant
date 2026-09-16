import io
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"diagnostics"))
from low_rank_recovery import attach_residual

class RecoveryContractTest(unittest.TestCase):
    def test_frozen_grid_gradient_and_restore(self):
        torch.manual_seed(7)
        base=torch.nn.Linear(8,6,bias=False)
        base.weight.data.zero_()
        teacher=torch.randn(6,2)@torch.randn(2,8)
        x=torch.randn(4,8)
        detail=attach_residual(base,teacher,2,7)
        torch.testing.assert_close(base(x),torch.nn.functional.linear(x,teacher),atol=1e-5,rtol=1e-5)
        self.assertLess(detail["residual_frobenius_unexplained_fraction"],1e-10)
        base(x).square().mean().backward()
        self.assertIsNone(base.weight.grad)
        self.assertGreater(base.awq_recovery_lora[0].weight.grad.abs().sum(),0)
        self.assertGreater(base.awq_recovery_lora[1].weight.grad.abs().sum(),0)
        self.assertEqual(base.weight.detach().abs().sum(),0)
        buf=io.BytesIO(); torch.save(base.state_dict(),buf); buf.seek(0)
        restored=torch.nn.Linear(8,6,bias=False)
        attach_residual(restored,torch.zeros_like(teacher),2,9)
        restored.load_state_dict(torch.load(buf,weights_only=True))
        torch.testing.assert_close(base(x),restored(x),atol=0,rtol=0)
        with self.assertRaises(ValueError): attach_residual(base,teacher,2,7)

if __name__=="__main__": unittest.main()
