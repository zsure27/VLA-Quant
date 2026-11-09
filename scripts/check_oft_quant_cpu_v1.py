"""Small official-source contracts; never loads an OFT checkpoint or searches it."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    sys.path.insert(0, str(args.code))
    import torch
    from qvla.official_quant_adapter import load_official_functions, apply_awq_entry
    result = {'gate': 'IN_PROGRESS_CPU_CONTRACTS', 'gpu_run_admitted': False}
    try:
        official = load_official_functions(args.root / 'sources/official-quantization-pinned-v2')
        torch.manual_seed(7)
        layer = torch.nn.Linear(128, 8).requires_grad_(False)
        expected = official['awq_quantize'](layer.weight.detach().clone(), n_bit=4, zero_point=True, q_group_size=128)
        apply_awq_entry(layer, {'input_scale': torch.ones(128), 'clip_max': None}, official, 4, 128)
        assert torch.equal(layer.weight, expected) and torch.isfinite(layer.weight).all()
        result['awq_wrapper_matches_fixed_official_quantizer'] = True
        norm = torch.nn.LayerNorm(16).requires_grad_(False)
        outputs = [torch.nn.Linear(16, 8).requires_grad_(False) for _ in range(3)]
        x = torch.randn(2, 3, 16)
        before = [module(norm(x)).detach().clone() for module in outputs]
        official['smooth_ln_fcs'](norm, outputs, torch.rand(16) + .1, alpha=.5)
        error = max(float((module(norm(x)) - saved).abs().max()) for module, saved in zip(outputs, before))
        assert error < 2e-5, error
        result['smoothquant_unquantized_transform_max_abs'] = error
        for bits in [4, 8]:
            y = official['smooth_activation_quantize'](x, n_bits=bits)
            assert y.shape == x.shape and torch.isfinite(y).all() and not y.is_cuda
        result['activation_w4a4_w4a8_finite_shapes'] = True
        result['scope'] = 'small CPU modules only; not model accuracy, full-model equivalence or native INT4 kernels'
        result['gate'] = 'PASS_OFT_FIXED_SOURCE_SMALL_CPU_CONTRACTS'
    except BaseException as error:
        result.update(gate='FAIL_OFT_SMALL_CPU_CONTRACTS', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
