"""Raw GR00T GPTQ pilot after separate artifact/reload admission, never a new formula."""
import argparse
import gc
import os
from pathlib import Path
import sys
import numpy as np
import torch
from qvla.groot_drift_20261113_v3 import load, infer, episode
from qvla.groot_snapshot_20261111_v2 import SUITES, verify_snapshot
from qvla.groot_sq_20261111_v2 import targets, read_obs
from qvla.pi05_packed_weight_v1 import unpack, weight_sha
from qvla.infra.common import digest, read_json, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--round-dir', type=Path, required=True)
    p.add_argument('--reference-round', type=Path, required=True)
    p.add_argument('--suite', choices=SUITES, required=True)
    p.add_argument('--profile', type=Path, required=True)
    a = p.parse_args()
    r = a.root.resolve()
    sys.path[:0] = [str(r / 'sources/Isaac-GR00T'), str(r / 'sources/LIBERO')]
    reference = a.reference_round / a.suite
    assert read_json(reference / 'reference-gate.json')['gate'] == 'PASS_GROOT_REFERENCE_REPEAT_AND_CALIBRATION'
    assert read_json(a.profile.parent / 'result.json')['profile_sha256'] == digest(a.profile)
    payload = torch.load(a.profile, map_location='cpu', weights_only=True)
    assert payload['format'] == 'GROOT_ORIGINAL_GPTQ_W4A16_V1'
    assert payload['gate'] == 'PASS_GROOT_GPTQ_RELOAD_ACTION' and payload['suite'] == a.suite
    verify_snapshot(r / 'models' / SUITES[a.suite][0], a.suite)
    assert payload['checkpoint_receipt_sha256'] == digest(r / 'models' / SUITES[a.suite][0] / 'DOWNLOAD_VERIFICATION.json')
    assert payload['reference_first_query_sha256'] == digest(reference / 'bf16/query-00000.npz')
    for sample in payload['samples']:
        assert sample['role'] == 'calibration' and sample['reset_index'] == 0
        path = Path(sample['path']).resolve()
        assert path.parent == (reference / 'calibration-reset0').resolve() and digest(path) == sample['sha256']
    policy = load(r, a.suite)
    names = targets(r, a.suite, policy.model)
    assert len(payload['entries']) == 116 and {x['name'] for x in payload['entries']} == set(names)
    modules = dict(policy.model.named_modules())
    for item in payload['entries']:
        path = (a.profile.parent / item['file']).resolve()
        assert a.profile.parent.resolve() in path.parents and digest(path) == item['sha256']
        saved = torch.load(path, map_location='cpu', weights_only=True)
        assert saved['name'] == item['name']
        layer = modules[item['name']]
        value = unpack(saved['entry'], layer.weight.device, layer.weight.dtype)
        assert value.shape == layer.weight.shape and weight_sha(value) == item['weight_sha256']
        with torch.no_grad():
            layer.weight.copy_(value)
    assert all(not x.requires_grad and torch.isfinite(x).all() for x in policy.model.parameters())
    parent = a.round_dir / a.suite
    parent.mkdir(exist_ok=True)
    write_json(parent / 'gptq-profile-context.json', dict(
        gate='PASS_GROOT_GPTQ_ARTIFACT_SCOPE_AND_PARENT', profile_sha256=digest(a.profile),
        target_count=116, source_sha256=payload['source_sha256'],
        recipe=payload['recipe'], actual_native_flow_shape=[1, 16, 32], actions_per_query=1,
        no_real_integer_kernel=True))
    # The frozen collector checks real first inputs and flow noise against BF16.
    episode(policy, r, a.suite, parent / 'gptq-w4a16', 5, 'development', reference=reference / 'bf16')
    del modules, policy
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
