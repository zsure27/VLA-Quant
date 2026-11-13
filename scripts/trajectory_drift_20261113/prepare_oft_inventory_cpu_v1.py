"""Streaming asset identity and exact existing-case inventory; never load 7B."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qvla.baseline_contract import canonical_targets


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    r, out = a.root.resolve(), a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    assert subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                          capture_output=True).returncode != 0, 'CPU-only authorization'
    model = r / 'models/openvla_oft'
    receipt_path = model / 'DOWNLOAD_VERIFICATION.json'
    receipt = json.loads(receipt_path.read_text())
    receipt_sha = sha(receipt_path)
    assert receipt_sha == '9a2f6e851c3be4ac54bb291ea539af7efc252487c1b6fdda950e685faa592edc'
    for entry in receipt['files_verified']:
        f = model / entry['path']
        assert f.stat().st_size == entry['bytes'] and sha(f) == entry['sha256'], str(f)
    print('MODEL25_STREAMING_SHA_PASS', flush=True)
    old = r / 'control/20261112-061-oft-qvla-resume-v6'
    summary_path = old / 'analysis-oft-seven-conditions-v2/summary.json'
    summary = json.loads(summary_path.read_text())
    assert summary['gate'] == 'PASS_OFT_7_CONDITIONS_80_MATCHED_CASES_480_PAIRS'
    profiles = {
        'awq_w4a16': r / 'profiles/20261109-four-suite-reset0-v4/awq.pt',
        'gptq_w4a16': r / 'profiles/20261110-gptq-full-v1/profile.pt',
        'omniquant_w4a16': r / 'profiles/20261110-omni-lwc-full-v2/profile.pt',
        'smoothquant_w4a8': r / 'profiles/20261109-four-suite-reset0-v4/smoothquant.pt',
        'smoothquant_w4a4': r / 'profiles/20261109-four-suite-reset0-v4/smoothquant.pt',
        'qvla_w4a16': r / 'control/20261112-061-oft-other-configs-v4/qvla-gates-v2.json',
    }
    digest_cache, actual_profiles, cases = {}, {}, {}
    def verified_case(path, condition):
        path = Path(path)
        z = json.loads(path.read_text())
        assert z['checkpoint_receipt_sha256'] == receipt_sha
        assert z['oft_source_commit'] == 'e4287e94541f459edc4feabc4e181f537cd569a8'
        assert z['condition'] == condition
        key = (condition, z['suite'], z['task_id'], z['initial_state_index'])
        record = dict(condition=condition, suite=z['suite'], task_id=z['task_id'],
                      reset_index=z['initial_state_index'], result=str(path), result_sha256=sha(path),
                      initial_state_sha256=z['initial_state_sha256'])
        if key in cases:
            assert cases[key] == record, 'Ambiguous canonical reference'
        cases[key] = record
        if condition != 'bf16':
            profile = profiles[condition]
            if profile not in digest_cache:
                digest_cache[profile] = sha(profile)
            assert digest_cache[profile] == z['profile_sha256']
            actual_profiles[condition] = dict(path=str(profile), bytes=profile.stat().st_size,
                                              sha256=digest_cache[profile])
        return record
    for row in summary['metrics']:
        verified_case(Path(row['candidate']) / 'result.json', row['condition'])
        verified_case(Path(row['reference']) / 'result.json', 'bf16')
    stage = old / 'stages/bf16-initial25-v2'
    series = json.loads((stage / 'series.json').read_text())
    for path, expected in series['external_frozen_sources'].items():
        assert sha(path) == expected
    audit_path = stage / 'reference-audit/result.json'
    audit = json.loads(audit_path.read_text())
    assert audit['gate'] == 'PASS_INITIAL25_BF16_RAW_REFERENCE_CASES' and audit['completed'] == 40
    for f in stage.glob('libero*-cases.json'):
        for row in json.loads(f.read_text())['completed']:
            assert sha(row['result_path']) == row['result_sha256']
            verified_case(row['result_path'], 'bf16')
    assert len(cases) == 600
    # Search ALL controls/runs, so an unindexed completed result cannot be rerun.
    all_results = set((r / 'runs').rglob('result.json'))
    for pattern in ['*/case-runs/*/result.json', '*/stages/*/case-runs/*/result.json']:
        all_results.update((r / 'control').glob(pattern))
    orphan25 = []
    for f in all_results:
        z = json.loads(f.read_text())
        if z.get('model') == 'openvla_oft' and z.get('initial_state_index') == 25 and z.get(
                'checkpoint_receipt_sha256') == receipt_sha and z.get('condition') in profiles:
            orphan25.append(str(f))
    assert not orphan25, 'Rebuild pending queue around existing candidate results'
    pending = []
    for condition in profiles:
        for suite in ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']:
            for task in range(10):
                reference = cases['bf16', suite, task, 25]
                pending.append(dict(condition=condition, suite=suite, task_id=task, reset_index=25,
                                    reference_result=reference['result'],
                                    reference_sha256=reference['result_sha256'],
                                    initial_state_sha256=reference['initial_state_sha256']))
    assert len(pending) == 240
    shapes = {}
    for f in model.glob('*.safetensors'):
        with f.open('rb') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            header = json.loads(stream.read(size))
        shapes.update({k: v['shape'] for k, v in header.items() if k != '__metadata__'})
    scope = []
    for name in canonical_targets():
        shape = shapes[name + '.weight']
        width = 1
        for size in shape[1:]:
            width *= size
        block = next(n for n in [128, 16, 4] if width % n == 0)
        scope.append(dict(name=name, weight_shape=shape, flattened_input_width=width,
                          rotation_block=block, kind='linear' if len(shape) == 2 else 'patch_conv'))
    assert len(scope) == 422
    result = dict(gate='PASS_CPU_ASSETS_600_IDENTITIES_AND_240_PENDING_INITIAL25',
                  observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  model_files_verified=len(receipt['files_verified']), model_bytes_verified=receipt['bytes'],
                  checkpoint_receipt_sha256=receipt_sha, profiles=actual_profiles,
                  complete_case_identities=600, repeated_controls_excluded=4,
                  global_result_files_checked=len(all_results), pending_initial25=240,
                  source_frozen_files_verified=series['external_frozen_sources'],
                  full_raw_reaudit_performed=False,
                  raw_evidence='Prior per-query/action/pose/video audits retained by SHA; current pass '
                               'rechecks complete result identities, all model bytes, profile bytes and sources.',
                  next_gpu_requires_new_auth=True, gpu_started=False, full_model_loaded=False)
    for name, value in [('result.json', result), ('pending-initial25.json', pending),
                        ('complete-case-identities.json', list(cases.values())), ('duquant-scope422.json', scope)]:
        (out / name).write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
