import sys
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"diagnostics"))
from scale_peft_recovery import attach_scale_peft
from qvla.fixed_code_scale import official_equivalent


def test_scale_peft_fixed_code_and_training_contract():
    torch.manual_seed(5)
    teacher=torch.randn(12,16)
    layer=torch.nn.Linear(16,12,bias=False)
    layer.weight.data.copy_(official_equivalent(teacher,2,8))
    baseline=layer.weight.detach().clone()
    x=torch.randn(24,16)
    detail=attach_scale_peft(layer,teacher,{"clip_max":None},2,8,x,30,1e-2)
    assert detail["zero_residual_exact"]
    assert detail["training_final_mse"] < detail["training_initial_mse"]
    assert torch.equal(layer.weight,baseline)
    assert layer.weight.grad is None
    assert torch.isfinite(layer.awq_scale_peft.log_step_residual).all()


def test_scale_peft_can_defer_training_to_end_to_end_objective():
    torch.manual_seed(6)
    teacher=torch.randn(12,16)
    layer=torch.nn.Linear(16,12,bias=False)
    layer.weight.data.copy_(official_equivalent(teacher,2,8))
    x=torch.randn(8,16)
    expected=layer(x).detach()
    detail=attach_scale_peft(layer,teacher,{"clip_max":None},2,8,x,0,1e-3)
    assert detail["training_initial_mse"] is None
    assert detail["training_final_mse"] is None
    assert torch.equal(layer(x),expected)
    assert layer.awq_scale_peft.log_step_residual.requires_grad
