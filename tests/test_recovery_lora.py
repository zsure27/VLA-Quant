import io

import torch

from qvla.recovery_lora import RecoveryLoRALinear


def test_standard_zero_output_gradient_and_reload():
    torch.manual_seed(11)
    source = torch.nn.Linear(64, 32, bias=False)
    module = RecoveryLoRALinear(source, rank=8, alpha=8)
    value = torch.randn(4, 64)
    assert torch.equal(module(value), source(value))
    loss = module(value).square().mean()
    loss.backward()
    assert module.base_weight.grad is None
    assert module.lora_A.grad is not None and torch.isfinite(module.lora_A.grad).all()
    assert module.lora_B.grad is not None and torch.isfinite(module.lora_B.grad).all()
    assert torch.count_nonzero(module.lora_A.grad) == 0
    assert torch.count_nonzero(module.lora_B.grad) > 0
    payload = io.BytesIO()
    torch.save(module.state_dict(), payload)
    payload.seek(0)
    clone = RecoveryLoRALinear(source, rank=8, alpha=8)
    clone.load_state_dict(torch.load(payload, weights_only=True))
    assert all(torch.equal(value, clone.state_dict()[key]) for key, value in module.state_dict().items())
