"""OFT-shaped CPU binding of pinned DuQuant primitives, not model admission.

This implements the optional no-LET/no-LWC core at explicit common input
boundaries. It does not implement the upstream full decoder-block optimizer or
attention-matmul quantization. Never register it as a completed paper recipe.
Rotations must be frozen from calibration data before evaluation.
"""
from pathlib import Path
import hashlib
import importlib.util
import sys
import types
import torch
from torch import nn
from torch.nn import functional as F

PINS = {
    'quantize/quantizer.py': '99c2a0d76866e2289658fb85a2b7e2673c23517fb1d9c68642753bf5b9830b1a',
    'utils.py': '9907095a77b674725118e2d0ff5041f34eb7e3576158c7df83c5638b7a6ad6dc',
}
FORMAT = 'OFT_DUQUANT_CORE_CPU_BINDING_V1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_original(source, rotations, expected_rotation_sha):
    """Import original functions with isolated absolute-import dependencies."""
    source = Path(source)
    for name, expected in PINS.items():
        if digest(source / name) != expected:
            raise ValueError('Pinned DuQuant source changed: ' + name)
    if digest(rotations) != expected_rotation_sha:
        raise ValueError('Rotation asset SHA differs')
    assets = torch.load(rotations, map_location='cpu', weights_only=True)
    previous = {name: sys.modules.get(name) for name in ['utils', 'quantize', 'quantize.const']}
    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        sys.modules[name] = value
        spec.loader.exec_module(value)
        return value
    try:
        utils = module('utils', source / 'utils.py')
        # Populate the original get_rot cache; no cwd-dependent pickle loading.
        utils.Rot.update(assets)
        package = types.ModuleType('quantize')
        package.__path__ = [str(source / 'quantize')]
        sys.modules['quantize'] = package
        module('quantize.const', source / 'quantize/const.py')
        quant = module('_oft_duquant_original', source / 'quantize/quantizer.py')
        return quant.UniformAffineQuantizer
    finally:
        sys.modules.pop('_oft_duquant_original', None)
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def validate_transform(rotation, permutation, width, block):
    if width % block or rotation.ndim != 3 or rotation.shape[1:] != (block, block):
        raise ValueError('Invalid width/rotation shape')
    if permutation.ndim != 2 or permutation.shape[1] != width or len(rotation) != len(permutation) + 1:
        raise ValueError('Invalid permutation shape')
    if not torch.isfinite(rotation).all():
        raise ValueError('Nonfinite rotation')
    identity = torch.eye(block, dtype=torch.float64)
    for matrix in rotation.double().cpu():
        if not torch.allclose(matrix.T @ matrix, identity, atol=3e-5, rtol=3e-5):
            raise ValueError('Nonorthogonal rotation')
    expected = torch.arange(width)
    for row in permutation.cpu():
        if row.dtype != torch.int64 or not torch.equal(row.sort().values, expected):
            raise ValueError('Permutation must contain every input channel once')


def calibrate_component(quantizer_class, rows, block, *, rotation_steps=4, role='calibration_reset0'):
    """Small CPU contract fixture; full model calibration is a separate stage."""
    if role != 'calibration_reset0' or rows.ndim != 2 or not torch.isfinite(rows).all():
        raise ValueError('Require finite calibration rows, never evaluation rows')
    if rows.shape[-1] % block:
        raise ValueError('Input width is not divisible by rotation block')
    quant = quantizer_class(n_bits=16, dynamic_method='per_token', quant_method='duquant',
                           block_size=block, rotate=True, max_rotation_step=rotation_steps,
                           permutation_times=1)
    quant.init_duquant(rows.detach().clone())
    result = dict(rotation=quant.R.detach().clone(), permutation=quant.permutation_list.detach().clone(),
                  width=rows.shape[-1], block=block, calibration_role=role,
                  rotation_steps=rotation_steps, source_sha256=PINS['quantize/quantizer.py'])
    validate_transform(result['rotation'], result['permutation'], result['width'], block)
    return result


class FrozenDuQuantLinear(nn.Module):
    """Preserves arbitrary leading dimensions; upstream core receives 2D rows."""
    def __init__(self, original, quantizer_class, profile, *, weight_bits=4, activation_bits=8):
        super().__init__()
        if weight_bits not in (4, 16) or activation_bits not in (4, 8, 16):
            raise ValueError('Unprepared precision')
        width = original.in_features
        if profile['width'] != width or profile['calibration_role'] != 'calibration_reset0':
            raise ValueError('Wrong input width or data role')
        if profile['source_sha256'] != PINS['quantize/quantizer.py']:
            raise ValueError('Wrong algorithm source')
        validate_transform(profile['rotation'], profile['permutation'], width, profile['block'])
        self.register_buffer('weight', original.weight.detach().clone())
        self.register_buffer('bias', original.bias.detach().clone() if original.bias is not None else None)
        self.in_features, self.out_features = width, original.out_features
        self.enabled = False
        self.identity = dict(format=FORMAT, width=width, block=profile['block'],
                             weight_bits=weight_bits, activation_bits=activation_bits,
                             calibration_role=profile['calibration_role'], source=PINS)
        for kind, bits, dynamic in [('weight', weight_bits, 'per_channel'),
                                     ('act', activation_bits, 'per_token')]:
            quant = quantizer_class(n_bits=bits, dynamic_method=dynamic, quant_method='duquant',
                                    block_size=profile['block'], rotate=True,
                                    shape=list(original.weight.shape) if kind == 'weight' else None,
                                    lwc=False)
            quant.R = profile['rotation'].detach().clone()
            quant.permutation_list = profile['permutation'].detach().clone()
            quant.init_duquant_params = torch.tensor(1)
            quant.register_duquant_params()
            self.add_module(kind + '_quantizer', quant)

    def forward(self, value):
        if value.shape[-1] != self.in_features:
            raise ValueError('Wrong input shape')
        if not self.enabled:
            return F.linear(value, self.weight, self.bias)
        shape = value.shape
        rows = value.reshape(-1, self.in_features).clone()
        activation = self.act_quantizer(rows)
        weight = self.weight_quantizer(self.weight.clone())
        return F.linear(activation, weight, self.bias).reshape(*shape[:-1], self.out_features)

    def get_extra_state(self):
        return self.identity

    def set_extra_state(self, state):
        if state != self.identity:
            raise ValueError('Source, precision or profile identity differs')

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if state_dict.get('_extra_state') != self.identity:
            raise ValueError('Source, precision or profile identity differs')
        for prefix in ['act_quantizer', 'weight_quantizer']:
            validate_transform(state_dict[prefix + '.R'], state_dict[prefix + '.permutation_list'],
                               self.in_features, self.identity['block'])
        if not torch.equal(state_dict['act_quantizer.R'], state_dict['weight_quantizer.R']) or not torch.equal(
                state_dict['act_quantizer.permutation_list'], state_dict['weight_quantizer.permutation_list']):
            raise ValueError('Weight and activation transforms must match')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


class FrozenDuQuantPatchConv(nn.Module):
    """Common-scope patch Conv2d extension, not upstream decoder-block code."""
    def __init__(self, original, quantizer_class, profile, **bits):
        super().__init__()
        if original.groups != 1 or original.padding_mode != 'zeros':
            raise ValueError('Only ungrouped zero-padded patch convolution prepared')
        self.kernel, self.stride = original.kernel_size, original.stride
        self.padding, self.dilation = original.padding, original.dilation
        linear = nn.Linear(original.weight[0].numel(), original.out_channels,
                           bias=original.bias is not None, device=original.weight.device,
                           dtype=original.weight.dtype)
        with torch.no_grad():
            linear.weight.copy_(original.weight.flatten(1))
            if original.bias is not None:
                linear.bias.copy_(original.bias)
        self.linear = FrozenDuQuantLinear(linear, quantizer_class, profile, **bits)

    def forward(self, value):
        patches = F.unfold(value, self.kernel, self.dilation, self.padding, self.stride).transpose(1, 2)
        result = self.linear(patches).transpose(1, 2)
        height, width = [
            (value.shape[i + 2] + 2 * self.padding[i] - self.dilation[i] * (self.kernel[i] - 1) - 1)
            // self.stride[i] + 1 for i in range(2)]
        return result.reshape(value.shape[0], self.linear.out_features, height, width)
