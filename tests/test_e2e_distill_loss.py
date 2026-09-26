import pytest
import torch

from diagnostics.probe import e2e_distill_loss


def test_smooth_l1_limits_outlier_dominance_and_keeps_gradients():
    prediction = torch.tensor([0.01, 2.0], requires_grad=True)
    target = torch.zeros_like(prediction)
    mse = e2e_distill_loss(prediction, target, "mse")
    robust = e2e_distill_loss(prediction, target, "smooth-l1", beta=0.1)
    assert robust < mse
    robust.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0


def test_unknown_loss_is_rejected():
    with pytest.raises(ValueError):
        e2e_distill_loss(torch.zeros(1), torch.zeros(1), "unknown")
