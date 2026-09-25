import io

import torch

from qvla.fixed_code_scale import (
    FixedCodeScaleLinear,
    attach_fixed_code_scale_state,
    build_fixed_code,
    dequantize_fixed_code,
    official_equivalent,
)


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


def test_attach_pretrained_scale_state_preserves_zero_contract_and_loads_residual():
    torch.manual_seed(11)
    source = torch.nn.Linear(128, 32, bias=False, dtype=torch.bfloat16)
    teacher = source.weight.detach().clone()
    source.weight.data.copy_(official_equivalent(teacher, 2, 64))
    residual = torch.randn(32 * 2, 1, dtype=torch.float32) * 1e-3
    details = attach_fixed_code_scale_state(source, teacher, {}, 2, 64, residual)
    assert details["parameters"] == residual.numel()
    assert torch.equal(source.awq_scale_peft.log_step_residual, residual)
    value = torch.randn(2, 128, dtype=torch.bfloat16)
    output = source(value)
    assert torch.isfinite(output).all()
