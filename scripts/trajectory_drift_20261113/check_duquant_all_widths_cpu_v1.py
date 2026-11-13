"""Complete actual OFT input-width coverage using small synthetic CPU fixtures."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from qvla.duquant_oft_core_v1 import (
    PINS, FrozenDuQuantLinear, calibrate_component, digest, load_original,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--control', type=Path, required=True)
    a = p.parse_args()
    r, control = a.root.resolve(), a.control.resolve()
    out = control / 'duquant-remaining-widths-cpu-v1'
    out.mkdir(exist_ok=False)
    assert not torch.cuda.is_available()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    asset = control / 'duquant-core-cpu-v1/rotations-weights-only.pt'
    cls = load_original(r / 'sources/DuQuant-archive-v1', asset, digest(asset))
    checks = []
    for width in [1024, 1152, 4096, 11008]:
        if width == 11008:
            saved = torch.load(control / 'duquant-core-cpu-v1/fixture-width11008-a8.pt', weights_only=True)
            profile = dict(width=width, block=128, calibration_role='calibration_reset0',
                           rotation=saved['act_quantizer.R'], permutation=saved['act_quantizer.permutation_list'],
                           source_sha256=PINS['quantize/quantizer.py'])
        else:
            rows = torch.randn(8, width)
            rows[:, 2] *= 30
            profile = calibrate_component(cls, rows, 128)
        layer, value = torch.nn.Linear(width, 9), torch.randn(2, 3, width)
        original = value.clone()
        all16 = FrozenDuQuantLinear(layer, cls, profile, weight_bits=16, activation_bits=16)
        assert torch.equal(all16(value), layer(value))
        all16.enabled = True
        error = float((all16(value) - layer(value)).abs().max())
        assert torch.allclose(all16(value), layer(value), atol=5e-4, rtol=5e-4)
        for bits in [4, 8]:
            module = FrozenDuQuantLinear(layer, cls, profile, activation_bits=bits)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                assert torch.equal(module(value.bfloat16()), layer(value.bfloat16()))
                module.enabled = True
                y = module(value.bfloat16())
                assert y.dtype == torch.bfloat16 and torch.isfinite(y).all()
            assert torch.equal(value, original)
        checks.append(dict(width=width, block=128, all16_float32_max_abs=error,
                           bf16_autocast_w4a4_w4a8_finite=True, no_original_input_mutation=True))
    old = json.loads((control / 'DUQUANT_BF16_CPU_DIAGNOSTIC.json').read_text())
    covered = {x['width'] for x in old['checks']} | {x['width'] for x in checks}
    scope = json.loads((control / 'inventory/duquant-scope422.json').read_text())
    actual = {x['flattened_input_width'] for x in scope}
    assert actual <= covered
    result = dict(gate='PASS_ALL_SIX_OFT_INPUT_WIDTHS_CPU_COMPONENTS',
                  actual_model_input_widths=sorted(actual), prior_diagnostic=old['gate'], checks=checks,
                  target_shape_entries=len(scope), actual_model_weights_loaded=False,
                  qualification='All actual input widths checked on small9-output synthetic modules; '
                                'not full matrices/422-layer injection, not GPU or native action admission',
                  gpu_started=False, model_admitted=False)
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
