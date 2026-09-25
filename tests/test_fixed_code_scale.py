import io

import torch

from qvla.fixed_code_scale import FixedCodeScaleLinear, build_fixed_code, dequantize_fixed_code, official_equivalent


def test_zero_residual_exact_and_gradient_reload():
    torch.manual_seed(7)
    source = torch.nn.Linear(128, 32, bias=False, dtype=torch.bfloat16)
    code = build_fixed_code(source.weight, 2, 64)
    assert torch.equal(dequantize_fixed_code(code), official_equivalent(source.weight, 2, 64))
    module = FixedCodeScaleLinear(source, 2, 64)
    loss = module(torch.randn(3, 128, dtype=torch.bfloat16)).float().square().mean()
    loss.backward()
    assert module.log_step_residual.grad is not None
    assert torch.isfinite(module.log_step_residual.grad).all()
    assert torch.count_nonzero(module.log_step_residual.grad)
    payload = io.BytesIO()
    torch.save(module.state_dict(), payload)
    payload.seek(0)
    clone = FixedCodeScaleLinear(source, 2, 64)
    clone.load_state_dict(torch.load(payload, weights_only=True))
    for key, value in module.state_dict().items():
        assert torch.equal(value, clone.state_dict()[key])
