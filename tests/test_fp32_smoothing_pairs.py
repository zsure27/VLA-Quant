import copy
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"diagnostics"))
from fp32_smoothing_pairs import promote_pairs

class PairContractTest(unittest.TestCase):
    def test_complete_pair_dtype_boundary_and_parameter_values(self):
        torch.manual_seed(7)
        model=torch.nn.Module()
        model.add_module("vision_backbone",torch.nn.Module())
        model.vision_backbone.add_module("norm",torch.nn.LayerNorm(8).bfloat16())
        model.vision_backbone.add_module("fc",torch.nn.Linear(8,6).bfloat16())
        ref=copy.deepcopy(model).float()
        x=torch.randn(3,8).bfloat16()
        expected=ref.vision_backbone.fc(ref.vision_backbone.norm(x.float())).bfloat16()
        handles,metadata=promote_pairs(model,[("vision_backbone.norm",["vision_backbone.fc"],"vision_backbone.fc")])
        actual=model.vision_backbone.fc(model.vision_backbone.norm(x))
        self.assertEqual(actual.dtype,torch.bfloat16)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertEqual(metadata[0]["fp32_parameter_elements"],70)
        for a,b in zip(model.parameters(),ref.parameters()): torch.testing.assert_close(a,b,rtol=0,atol=0)
        for h in handles: h.remove()

if __name__=="__main__": unittest.main()
