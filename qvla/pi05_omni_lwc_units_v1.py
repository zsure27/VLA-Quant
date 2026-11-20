"""Original OmniQuant LWC on native attention/MLP units, not a full LET port.

Only clipping parameters train. The eval teacher and its mixed dtypes stay
unchanged. Prefix KV is copied per replay, including gradient-enabled replay.
"""
import contextlib
import copy
import importlib
from pathlib import Path
import sys
import types

import numpy as np
import torch
from torch import nn

from qvla.infra.common import digest
from qvla.pi05_block_replay_v2 import capture_call, LayerKVCache, _copy_tree
from qvla.pi05_awq_block_calibration_v1 import unique_tensor_bytes, registered_group
from qvla.pi05_packed_weight_v1 import pack, unpack, weight_sha
from qvla.pi05_torch_backend_v2 import target_names, BACKEND_ID

SOURCE_SHA = {'quantize/int_linear.py': '88440aba24a850bfa858ea879b7ebdc4f5f594276622499eff932a3d87a59b14',
              'quantize/quantizer.py': 'df933fc772b97bfee5f48d7fb3b47ab6ebea963ed282c7edea06f94ccb5829dd'}


def original_linear(source):
    source = Path(source).resolve()
    for name, sha in SOURCE_SHA.items():
        if digest(source / name) != sha:
            raise ValueError('Original OmniQuant source differs: ' + name)
    if 'quantize' not in sys.modules:
        package = types.ModuleType('quantize')
        package.__path__ = [str(source / 'quantize')]
        sys.modules['quantize'] = package
    module = importlib.import_module('quantize.int_linear')
    quantizer = importlib.import_module('quantize.quantizer')
    if Path(module.__file__).resolve() != source / 'quantize/int_linear.py' or Path(quantizer.__file__).resolve() != source / 'quantize/quantizer.py':
        raise ValueError('Quantizer import namespace collision')
    return module.QuantLinear


def tensor(value):
    value = value[0] if isinstance(value, tuple) else value
    if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
        raise ValueError('Unsupported/nonfinite native unit output')
    return value


def replay(module, captured):
    """Same native replay as v2; gradient state is owned by this caller."""
    if module.training or captured['schema_version'] != 1:
        raise ValueError('Frozen eval-mode native unit required')
    device = next(module.buffers(), None)
    parameter = next(module.parameters(), None)
    device = parameter.device if parameter is not None else device.device
    kwargs = _copy_tree(captured['kwargs'], device)
    if captured['cache_argument_supplied']:
        pair = _copy_tree(captured['cache_pair'], device)
        kwargs['past_key_value'] = LayerKVCache(captured['layer_index'], pair) if captured['cache_present'] else None
    return tensor(module(_copy_tree(captured['hidden'], device), **kwargs))


def inventory(model):
    targets = set(target_names(model))
    blocks = sorted({name.rsplit('.', 2)[0] for name in targets})
    units = []
    seen = set()
    for block in blocks:
        for suffix in ('self_attn', 'mlp'):
            name = block + '.' + suffix
            module = model.get_submodule(name)
            names = {name + '.' + key for key, child in module.named_modules() if isinstance(child, nn.Linear)}
            if not names or not names <= targets or seen & names:
                raise ValueError('Native unit target scope differs')
            seen |= names
            units.append((name, sorted(names)))
    if len(blocks) != 63 or len(units) != 126 or seen != targets or len(seen) != 414:
        raise ValueError('Require all native 126 units / 414 target projections')
    return units


@torch.no_grad()
def capture(policy, module, rows, max_bytes=2 * 1024**3):
    import inspect
    if len(rows) != 32 or len({r['sha256'] for r in rows}) != 32 or len({r['suite'] for r in rows}) != 4:
        raise ValueError('Fixed 32-query four-suite calibration required')
    if module.training or any(p.requires_grad for p in policy._model.parameters()):
        raise ValueError('Original eval teacher must be frozen')
    calls, teachers = [], []
    def before(unit, args, kwargs):
        bound = dict(inspect.signature(unit.forward).bind(*args, **kwargs).arguments)
        key = next((k for k in ('hidden_states', 'input', 'x') if k in bound), None)
        if key is None:
            raise ValueError('Unknown native unit signature')
        hidden = bound.pop(key)
        for n, parameter in inspect.signature(unit.forward).parameters.items():
            if parameter.kind == inspect.Parameter.VAR_KEYWORD and n in bound:
                extra = bound.pop(n)
                if set(extra) & set(bound):
                    raise ValueError('Ambiguous captured keyword')
                bound.update(extra)
        calls.append(capture_call((hidden,), bound, getattr(unit, 'layer_idx', 0)))
    def after(unit, args, value):
        teachers.append(tensor(value).detach().cpu().clone())
        if unique_tensor_bytes((calls, teachers)) > max_bytes:
            raise ValueError('Native unit calibration exceeds fixed 2GiB budget')
    hooks = [module.register_forward_pre_hook(before, with_kwargs=True), module.register_forward_hook(after)]
    versions = {n: p._version for n, p in policy._model.named_parameters()}
    try:
        for row in rows:
            path = Path(row['path'])
            if row['role'] != 'calibration' or row['reset_index'] != 0 or digest(path) != row['sha256']:
                raise ValueError('Calibration role or SHA differs')
            with np.load(path, allow_pickle=False) as q:
                obs = {'observation/image': q['image'], 'observation/wrist_image': q['wrist_image'],
                       'observation/state': q['state'], 'prompt': str(q['prompt'].item())}
                result = np.asarray(policy.infer(obs, noise=q['policy_noise'])['actions'])
                if result.shape != (10, 7) or not np.isfinite(result).all() or not np.array_equal(result, q['raw_policy_chunk']):
                    raise ValueError('Hooked teacher action differs from frozen BF16 query')
    finally:
        for hook in hooks:
            hook.remove()
    if not calls or len(calls) != len(teachers) or versions != {n: p._version for n, p in policy._model.named_parameters()}:
        raise ValueError('Unit was not called or teacher parameters changed')
    return calls, teachers


def calibrate(module, calls, teachers, source, *, epochs=20, lr=.01, fixture_group=None):
    if module.training or len(calls) != len(teachers) or not calls or any(p.requires_grad for p in module.parameters()):
        raise ValueError('Frozen native eval unit/captures required')
    original = {name: weight_sha(p) for name, p in module.named_parameters()}
    quantized = copy.deepcopy(module).eval()
    cls = original_linear(source)
    wrapped = {}
    for name, linear in list(quantized.named_modules()):
        if not isinstance(linear, nn.Linear):
            continue
        group = fixture_group or registered_group(linear.in_features)
        params = dict(n_bits=4, symmetric=False, per_channel_axes=[0], dynamic_method='per_channel',
                      group_size=group, lwc=True)
        wrapper = cls(linear, weight_quant_params=params, disable_input_quant=True).to(linear.weight.device).eval()
        parent, _, leaf = name.rpartition('.')
        setattr(quantized.get_submodule(parent) if parent else quantized, leaf, wrapper)
        wrapped[name] = wrapper
    if not wrapped:
        raise ValueError('No native linear projections')
    device = next(quantized.parameters()).device
    context = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else contextlib.nullcontext()
    with torch.no_grad(), context():
        for call, expected in zip(calls, teachers):
            if not torch.equal(replay(quantized, call).cpu(), expected):
                raise ValueError('Disabled quantization changed native unit output')
    for wrapper in wrapped.values():
        wrapper.set_quant_state(True, False)
    parameters = [p for p in quantized.parameters() if p.requires_grad]
    if not parameters or any('bound_factor' not in n for n, p in quantized.named_parameters() if p.requires_grad):
        raise ValueError('Only original LWC bound factors may train')
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0)
    losses, nonzero = [], False
    for _ in range(epochs):
        values = []
        for call, expected in zip(calls, teachers):
            optimizer.zero_grad(set_to_none=True)
            with context():
                value = replay(quantized, call)
                loss = (value.float() - expected.to(device).float()).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite LWC loss')
            loss.backward()
            gradients = [p.grad for p in parameters]
            if any(g is None or not torch.isfinite(g).all() for g in gradients):
                raise ValueError('Missing/nonfinite LWC gradient')
            nonzero |= any(bool(g.abs().sum() > 0) for g in gradients)
            optimizer.step()
            values.append(float(loss.detach()))
        losses.append(sum(values) / len(values))
    if not nonzero:
        raise ValueError('No nonzero LWC gradient')
    entries = {}
    with torch.no_grad():
        for name, wrapper in wrapped.items():
            q = wrapper.weight_quantizer
            value = q(wrapper.weight).to(wrapper.weight.dtype)
            rows, cols = wrapper.weight.shape
            groups = cols // q.group_size
            scales = q.scale.reshape(rows, groups)
            zeros = q.round_zero_point.reshape(rows, groups)
            entry = pack(value, [scales[:, i] for i in range(groups)], [zeros[:, i] for i in range(groups)], q.group_size)
            entry.update(lwc_upbound_factor=q.upbound_factor.detach().cpu(), lwc_lowbound_factor=q.lowbound_factor.detach().cpu())
            if not torch.equal(unpack(entry, value.device, value.dtype), value):
                raise ValueError('LWC packing/reload changed weight')
            entries[name] = entry
    if original != {name: weight_sha(p) for name, p in module.named_parameters()}:
        raise ValueError('Original teacher unit was modified')
    return dict(weights=entries, epochs=epochs, learning_rate=lr, weight_decay=0, LET=False,
                disabled_quant_bitwise=True, gradient_finite_nonzero=True, teacher_unchanged=True,
                native_unit_reconstruction=True, loss_first=losses[0], loss_last=losses[-1], calls=len(calls))


@torch.no_grad()
def apply_profile(model, payload, directory, *, allow_pending=False):
    if payload.get('format') != 'PI05_ORIGINAL_OMNI_LWC_NATIVE_UNITS_V1' or payload.get('backend') != BACKEND_ID:
        raise ValueError('Wrong OmniQuant profile format/backend')
    if not allow_pending and (payload.get('full_policy_admitted') is not True or payload.get('gate') != 'PASS_PI05_OMNI_LWC_NATIVE_414_RELOAD_FINITE'):
        raise ValueError('Full actual GPU admission is required')
    expected = dict(inventory(model))
    seen = set()
    if len(payload['entries']) != 126:
        raise ValueError('Partial unit profile rejected')
    for row in payload['entries']:
        name = row['name']
        path = Path(directory) / row['file']
        if name not in expected or path.is_symlink() or digest(path) != row['sha256']:
            raise ValueError('Native unit artifact identity differs')
        artifact = torch.load(path, map_location='cpu', weights_only=True)
        names = {name + '.' + n for n in artifact['weights']}
        if names != set(expected[name]) or names & seen:
            raise ValueError('Native unit weight scope differs')
        for relative, entry in artifact['weights'].items():
            linear = model.get_submodule(name + '.' + relative)
            weight = unpack(entry, linear.weight.device, linear.weight.dtype)
            if not torch.isfinite(weight).all() or weight_sha(weight) != entry['weight_sha256']:
                raise ValueError('Native LWC weight reload differs')
            linear.weight.copy_(weight)
        seen |= names
    if seen != set(target_names(model)) or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Full target scope/frozen base contract failed')
