"""CPU-only original DuQuant core checks at OFT input dimensions; no model load."""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from qvla.duquant_oft_core_v1 import (
    FrozenDuQuantLinear, FrozenDuQuantPatchConv, calibrate_component, digest, load_original,
)


def rejected(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('Invalid profile was admitted')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    assert not torch.cuda.is_available()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    source = (a.source / 'get_rot.py').read_bytes()
    gitsha = lambda value: hashlib.sha1(b'blob ' + str(len(value)).encode() + b'\0' + value).hexdigest()
    expected = 'ba28805a63bf18cee2bcc23ab042b1c38a84ad15'
    if gitsha(source) != expected and gitsha(source.rstrip(b'\n')) == expected:
        source = source.rstrip(b'\n')
    assert gitsha(source) == expected
    tree = ast.parse(source)
    device = next(x for x in tree.body if isinstance(x, ast.Assign) and isinstance(x.targets[0], ast.Name)
                  and x.targets[0].id == 'device')
    device.value = ast.Constant(value='cpu')
    loop = next(x for x in tree.body if isinstance(x, ast.For))
    loop.iter = ast.List(elts=[ast.Constant(value=n) for n in [4, 16, 128]], ctx=ast.Load())
    ast.fix_missing_locations(tree)
    cwd = Path.cwd()
    try:
        os.chdir(a.output)
        exec(compile(tree, 'original-get_rot[CPU,4/16/128]', 'exec'), {'__name__': '__main__'})
        assets = pickle.load(open('Rot.pkl', 'rb'))  # Own freshly generated, pinned-source output only.
        torch.save(assets, 'rotations-weights-only.pt')
    finally:
        os.chdir(cwd)
    asset = a.output / 'rotations-weights-only.pt'
    quantizer = load_original(a.source, asset, digest(asset))
    checks = []
    for width, block in [(128, 128), (4304, 16), (11008, 128), (588, 4), (768, 128)]:
        data = torch.randn(8, width)
        data[:, 1] *= 40
        profile = calibrate_component(quantizer, data, block, rotation_steps=4)
        layer = torch.nn.Linear(width, 9)
        original_weight = layer.weight.detach().clone()
        value = torch.randn(2, 3, width)
        before = value.clone()
        reference = layer(value)
        identity = FrozenDuQuantLinear(layer, quantizer, profile, weight_bits=16, activation_bits=16)
        assert torch.equal(identity(value), reference)
        identity.enabled = True
        transformed = identity(value)
        error = float((transformed - reference).abs().max())
        assert torch.allclose(transformed, reference, atol=5e-4, rtol=5e-4)
        results = {}
        for bits in [8, 4]:
            module = FrozenDuQuantLinear(layer, quantizer, profile, activation_bits=bits)
            module.enabled = True
            output = module(value)
            # Compare to unmodified original primitives on explicit flattened rows.
            expected_output = torch.nn.functional.linear(
                module.act_quantizer(value.reshape(-1, width).clone()),
                module.weight_quantizer(layer.weight.detach().clone()), layer.bias).reshape(2, 3, 9)
            assert torch.equal(output, expected_output) and torch.isfinite(output).all()
            state = module.state_dict()
            file = a.output / f'fixture-width{width}-a{bits}.pt'
            torch.save(state, file)
            clone = FrozenDuQuantLinear(layer, quantizer, profile, activation_bits=bits)
            clone.load_state_dict(torch.load(file, weights_only=True))
            clone.enabled = True
            assert torch.equal(clone(value), output)
            bad = copy.deepcopy(state)
            bad['act_quantizer.permutation_list'][0, 0] = bad['act_quantizer.permutation_list'][0, 1]
            rejected(lambda: clone.load_state_dict(bad))
            wrong = FrozenDuQuantLinear(layer, quantizer, profile, activation_bits=4 if bits == 8 else 8)
            rejected(lambda: wrong.load_state_dict(state))
            assert torch.equal(value, before) and torch.equal(layer.weight, original_weight)
            results[str(bits)] = dict(original_formula_bitwise=True, save_reload_bitwise=True,
                                     invalid_permutation_rejected=True, wrong_precision_rejected=True)
        wrong_role = dict(profile, calibration_role='evaluation_initial5')
        rejected(lambda: FrozenDuQuantLinear(layer, quantizer, wrong_role))
        rejected(lambda: calibrate_component(quantizer, data, block, role='evaluation_initial5'))
        checks.append(dict(input_width=width, rotation_block=block, leading_shape=[2, 3],
                           quant_disabled_bitwise=True, all16_transform_max_abs=error,
                           weight4_activation_results=results, input_and_original_weight_unchanged=True))
    patch_checks = []
    for kernel, block in [(14, 4), (16, 128)]:
        conv = torch.nn.Conv2d(3, 6, kernel, stride=kernel, bias=True)
        rows = torch.randn(8, 3 * kernel * kernel)
        profile = calibrate_component(quantizer, rows, block, rotation_steps=4)
        image = torch.randn(2, 3, kernel * 2, kernel * 3)
        wrapper = FrozenDuQuantPatchConv(conv, quantizer, profile)
        error = float((conv(image) - wrapper(image)).abs().max())
        assert torch.allclose(conv(image), wrapper(image), atol=2e-6, rtol=2e-5)
        for bits in [8, 4]:
            wrapper = FrozenDuQuantPatchConv(conv, quantizer, profile, activation_bits=bits)
            wrapper.linear.enabled = True
            assert torch.isfinite(wrapper(image)).all() and wrapper(image).shape == conv(image).shape
        patch_checks.append(dict(kernel=kernel, width=3 * kernel * kernel, block=block,
                                  unquantized_unfold_max_abs=error, w4a8_w4a4_finite=True))
    result = dict(gate='PASS_DUQUANT_OFT_CORE_SHAPES_RELOAD_AND_ROLE_CPU',
                  original_commit='d56cfc6fe97c34c0eb100fec82fe439865905679',
                  source_sha256=digest(a.source / 'quantize/quantizer.py'),
                  generator_git_blob=expected, rotation_asset_sha256=digest(asset),
                  binding_sha256=digest(Path(__file__).resolve().parents[2] / 'qvla/duquant_oft_core_v1.py'),
                  checks=checks, patch_checks=patch_checks, model_loaded=False, gpu_started=False,
                  qualification='Core primitives only; no LET/LWC or attention-matmul path; '
                                '4 fixture rotation steps, not final calibration256; '
                                'no full422 runtime injection/calibration/native action admission',
                  model_admitted=False)
    (a.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
