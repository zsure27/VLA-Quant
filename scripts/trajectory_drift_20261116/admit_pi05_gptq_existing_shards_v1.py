"""Admit unchanged original-GPTQ shards under the native backend math contract.

No quantization or calibration is rerun. Original failed artifacts are read-only.
This runs only at a GPU stage boundary, one policy at a time. Exact action replay
and exact weight roundtrip remain mandatory; no tolerance is introduced.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from qvla.infra.common import digest, write_json, now
from qvla.pi05_torch_backend_v2 import load_policy, target_names, BACKEND_ID
from qvla.pi05_packed_weight_v1 import unpack, weight_sha

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--parent-profile', type=Path, required=True)
p.add_argument('--capture', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
started = time.monotonic()
source = a.root / 'sources/openpi/src/openpi/models_pytorch/pi0_pytorch.py'
assert digest(source) == 'dcd7b6f508808ce4b7170a1720a4c62c6da04edae3652d16c94a469dca924e16'
assert 'torch.set_float32_matmul_precision("high")' in source.read_text()
progress = json.loads((a.parent_profile / 'progress.json').read_text())
assert progress['completed_layers'] == progress['target_layers'] == 414
entries = progress['entries']
assert len(entries) == 414 and len({x['name'] for x in entries}) == 414
capture = json.loads((a.capture / 'result.json').read_text())
assert capture['gate'] == 'PASS_PI05_414_CALIBRATION_REPLAY_STATS'
assert digest(a.capture / 'statistics.pt') == capture['profile_sha256']
cache = torch.load(a.capture / 'statistics.pt', map_location='cpu', weights_only=True)
assert cache['backend'] == BACKEND_ID
samples = {}
for row in cache['samples']:
    assert row['role'] == 'calibration' and row['reset_index'] == 0
    samples.setdefault(row['suite'], row)
assert len(samples) == 4
a.output.mkdir(exist_ok=False, parents=True)
(a.output / 'layers').mkdir()
for row in entries:
    old = a.parent_profile / row['file']
    assert old.is_file() and digest(old) == row['sha256']
    new = a.output / row['file']
    assert new.parent == a.output / 'layers' and not new.exists()
    os.link(old, new)  # Immutable bytes shared on the same persistent filesystem.
    assert digest(new) == row['sha256']

def flags():
    return dict(matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
                float32_precision=torch.get_float32_matmul_precision())

def load_unchanged_quantized():
    policy = load_policy(a.root)
    native = flags()
    assert native['float32_precision'] == 'high' and native['matmul_allow_tf32']
    modules = dict(policy._model.named_modules())
    assert set(target_names(policy._model)) == {x['name'] for x in entries}
    for row in entries:
        item = torch.load(a.output / row['file'], map_location='cpu', weights_only=True)
        assert item['name'] == row['name']
        module = modules[row['name']]
        weight = unpack(item['entry'], module.weight.device, module.weight.dtype)
        assert weight_sha(weight) == row['weight_sha256']
        assert torch.isfinite(weight).all()
        with torch.no_grad():
            module.weight.copy_(weight)
    assert all(not x.requires_grad for x in policy._model.parameters())
    assert all(torch.isfinite(x).all().item() for x in policy._model.parameters())
    assert flags() == native
    return policy, native

def action(policy, row):
    assert digest(row['path']) == row['sha256']
    with np.load(row['path'], allow_pickle=False) as q:
        obs = {'observation/image': q['image'], 'observation/wrist_image': q['wrist_image'],
               'observation/state': q['state'], 'prompt': str(q['prompt'].item())}
        noise = q['policy_noise']
    value = np.asarray(policy.infer(obs, noise=noise)['actions'])
    assert value.shape == (10, 7) and np.isfinite(value).all()
    return value.copy()

policy, first_flags = load_unchanged_quantized()
expected = [torch.from_numpy(action(policy, row)) for row in samples.values()]
assert all(np.array_equal(action(policy, row), v.numpy()) for row, v in zip(samples.values(), expected))
# Diagnostic only: emulate the mismatched first-inference state of the failed
# producer, then restore the native constructor contract before any admission.
torch.backends.cuda.matmul.allow_tf32 = False
legacy = [action(policy, row) for row in samples.values()]
torch.set_float32_matmul_precision('high')
assert flags() == first_flags
assert all(np.array_equal(action(policy, row), v.numpy()) for row, v in zip(samples.values(), expected))
counterfactual = [dict(suite=row['suite'], exactly_equal=bool(np.array_equal(x, y.numpy())),
                       max_abs=float(np.max(np.abs(x-y.numpy()))),
                       rmse=float(np.sqrt(np.mean((x.astype('float64')-y.numpy())**2))))
                  for row, x, y in zip(samples.values(), legacy, expected)]
np.savez_compressed(a.output / 'matmul-state-counterfactual.npz',
                    native=np.stack([x.numpy() for x in expected]), legacy=np.stack(legacy))
del policy
gc.collect()
torch.cuda.empty_cache()
policy, second_flags = load_unchanged_quantized()
assert first_flags == second_flags
reload_checks = []
for row, target in zip(samples.values(), expected):
    actual = action(policy, row)
    exact = bool(np.array_equal(actual, target.numpy()))
    reload_checks.append(dict(suite=row['suite'], exact=exact,
                              max_abs=float(np.max(np.abs(actual-target.numpy())))))
    assert exact, 'Native-contract reloaded action differs; artifact remains unadmitted'
payload = dict(format='PI05_ORIGINAL_GPTQ_PACK_V1', backend=BACKEND_ID,
               checkpoint_sha256=cache['checkpoint_sha256'],
               capture_sha256=capture['profile_sha256'], entries=entries,
               calibration_probe_samples=list(samples.values()),
               calibration_expected_actions={'16': expected},
               precision_gates={'16': 'PASS_RELOAD_FINITE'},
               recipe='Unchanged original GPTQ shards: reservoir128, groups128 ragged, damp0.01, block128, actorderFalse; no sequential propagation',
               execution_contract=first_flags,
               parent_failed_progress_sha256=digest(a.parent_profile / 'progress.json'),
               quantization_rerun=False, gate='PASS_PI05_GPTQ_414_PACK_RELOAD')
torch.save(payload, a.output / 'profile.pt')
write_json(a.output / 'result.json', dict(
    gate=payload['gate'], targets=414, precision_gates=payload['precision_gates'],
    profile_sha256=digest(a.output / 'profile.pt'),
    parent_failed_profile=str(a.parent_profile),
    original_shards_unchanged=True, quantization_rerun=False,
    native_first_flags=first_flags, native_reloaded_flags=second_flags,
    four_calibration_same_input_repeat_and_reload_exact=reload_checks,
    legacy_tf32_counterfactual=counterfactual,
    original_quantization_cost_separate=True,
    wall_seconds=time.monotonic()-started, time_utc=now()))
print('PASS_PI05_GPTQ_414_PACK_RELOAD_UNCHANGED_SHARDS_NATIVE_MATMUL')
