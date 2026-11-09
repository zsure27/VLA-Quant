"""CPU-only OFT asset/profile audit and blocked continuation inventories."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, code, out = args.root.resolve(), args.code.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    r = {'gate': 'IN_PROGRESS_CPU_ONLY', 'gpu_run_admitted': False,
         'observed_at': datetime.now(timezone.utc).isoformat(), 'sources': {}}
    try:
        for name, pin in [('openvla-oft', 'e4287e94541f459edc4feabc4e181f537cd569a8'),
                          ('LIBERO', '8f1084e3132a39270c3a13ebe37270a43ece2a01')]:
            head = subprocess.check_output(['git', '-C', str(root / 'sources' / name), 'rev-parse', 'HEAD'], text=True).strip()
            assert head == pin
            r['sources'][name] = head
        model = root / 'models/openvla_oft'
        receipt = model / 'DOWNLOAD_VERIFICATION.json'
        rows = json.loads(receipt.read_text())
        hashes, tensors = {}, {}
        for row in rows['files_verified']:
            rel = Path(row['path'])
            assert not rel.is_absolute() and '..' not in rel.parts
            p = model / rel
            assert p.is_file() and not p.is_symlink() and p.stat().st_size == row['bytes']
            hashes[row['path']] = sha(p)
            assert hashes[row['path']] == row['sha256'], str(p)
            if p.suffix == '.safetensors':
                with p.open('rb') as stream:
                    n = int.from_bytes(stream.read(8), 'little')
                    assert 0 < n < 64 * 1024 * 1024
                    header = json.loads(stream.read(n))
                tensors.update({k: v for k, v in header.items() if k != '__metadata__'})
        assert sum(x['bytes'] for x in rows['files_verified']) == rows['bytes']
        r['checkpoint'] = {'files_verified': len(hashes), 'bytes': rows['bytes'],
                           'receipt_sha256': sha(receipt), 'repo': rows['repo'], 'revision': rows['revision']}
        print('PASS_OFT_CHECKPOINT_SHA', len(hashes), rows['bytes'], flush=True)
        sys.path.insert(0, str(code))
        from qvla.baseline_contract import canonical_targets, check_scope, CONTRACT_ID
        names = (root / 'checks/connected422-20261109.txt').read_text().splitlines()
        names = [x.strip() for x in names if x.strip()]
        check_scope(names)
        assert set(names) == set(canonical_targets())
        r['target_inventory'] = {'count': len(names), 'tensor_count': len(tensors),
                                  'targets_file_sha256': sha(root / 'checks/connected422-20261109.txt')}
        (out / 'target-inventory.json').write_text(json.dumps({'targets': sorted(names),
              'checkpoint_tensor_shapes': {k: v['shape'] for k, v in tensors.items()}}, indent=2))
        calibration = root / 'calibration/20261109-four-suite-reset0-v1'
        manifest = json.loads((calibration / 'manifest.json').read_text())
        assert manifest['gate'] == 'PASS_DISTINCT_RESET0_CALIBRATION_CAPTURE'
        samples = {}
        for sample in manifest['samples']:
            assert sample['role'] == 'quant_calibration' and sample['reset_index'] == 0
            assert sha(calibration / sample['path']) == sample['sha256']
            samples[sample['path']] = sample['sha256']
        assert len(samples) == 4
        r['calibration'] = {'sample_sha256': samples, 'manifest_sha256': sha(calibration / 'manifest.json'),
                             'evaluation_resets_disjoint': [5, 15, 25, 35, 45]}
        import torch
        r['profiles'] = {}
        for method in ['awq', 'smoothquant']:
            path = root / 'profiles/20261109-four-suite-reset0-v4' / (method + '.pt')
            payload = torch.load(path, map_location='cpu', weights_only=True)
            assert payload['format_version'] == 3 and payload['method'] == method and payload['contract_id'] == CONTRACT_ID
            assert set(payload['entries']) == set(names)
            assert payload['sample_sha256'] == samples
            for file, expected in payload['checkpoint_identity']['files'].items():
                actual = hashes.get(file) or sha(model / file)
                assert actual == expected, file
            count = 0
            def finite(value):
                nonlocal count
                if isinstance(value, torch.Tensor):
                    assert not value.is_cuda and torch.isfinite(value).all()
                    count += 1
                elif isinstance(value, dict):
                    for x in value.values(): finite(x)
                elif isinstance(value, (list, tuple)):
                    for x in value: finite(x)
            finite(payload)
            if method == 'awq':
                assert payload['algorithm'] == 'awq_official_llama_vision_adapter_v1'
            r['profiles'][method] = {'sha256': sha(path), 'bytes': path.stat().st_size,
                'entries': len(payload['entries']), 'finite_tensors': count,
                'checkpoint_identity': payload['checkpoint_identity']['sha256'],
                'algorithm': payload.get('algorithm'), 'scope': 'CPU identity and finite audit; no new full-model admission'}
            del payload
            gc.collect()
        completed, duplicates = {}, []
        for path in sorted((root / 'runs').glob('*/result.json')):
            row = json.loads(path.read_text())
            if row.get('model') != 'openvla_oft' or not row.get('gate', '').startswith('PASS'):
                continue
            key = (row['condition'], row['suite'], row['task_id'], row.get('initial_state_index', row.get('reset_index')))
            if key[-1] not in [5, 15, 25, 35, 45]:
                continue
            if key in completed:
                duplicates.append(str(path.relative_to(root)))
                continue
            completed[key] = {'condition': key[0], 'suite': key[1], 'task_id': key[2], 'reset_index': key[3],
                               'result': str(path.relative_to(root)), 'result_sha256': sha(path), 'success': row['success']}
        r['reference_inventory'] = {'counts_by_condition': dict(Counter(x[0] for x in completed)),
                                    'duplicates_excluded': duplicates, 'completed_cases': list(completed.values()),
                                    'scope': 'result metadata; original raw audit remains authoritative'}
        (out / 'reference-inventory.json').write_text(json.dumps(r['reference_inventory'], indent=2))
        catalog = json.loads((code / 'configs/infra/runtime_commonality_v6/method_catalog.json').read_text())
        for suite in ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']:
            for i, item in enumerate([{'method': 'BF16', 'precision': 'BF16'}, *catalog['methods']]):
                cfg = {'model': 'openvla_oft', 'suite': suite, 'method': item['method'], 'precision': item['precision'],
                       'tasks': list(range(10)), 'initial_states': [5, 15, 25, 35, 45], 'native_execute_horizon': 8,
                       'executable': False, 'gpu_run_admitted': False, 'checkpoint_receipt_sha256': sha(receipt),
                       'existing_profiles': r['profiles'] if item['method'] in ['AWQ', 'SmoothQuant'] else {},
                       'skip_completed_only_after_exact_condition_and_artifact_identity_match': True,
                       'remaining_gate': 'fresh GPU authorization, exact source/profile recipe and numerical admission; unsupported bindings stay blocked'}
                (out / f'{suite}-{i:02d}-{item["method"].lower()}-{item["precision"].lower()}.json').write_text(json.dumps(cfg, indent=2))
        r['blocked_templates'] = 52
        r['environment'] = {'hostname': subprocess.check_output(['hostname'], text=True).strip(),
              'disk_free_bytes': shutil.disk_usage(root).free, 'memory_max': Path('/sys/fs/cgroup/memory.max').read_text().strip(),
              'code_path': str(code), 'code_commit': subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip()}
        r['package_versions'] = {x: importlib.metadata.version(x) for x in ['torch', 'transformers', 'numpy', 'mujoco', 'robosuite']}
        r['gate'] = 'PASS_OFT_CPU_ASSET_PROFILE_AND_METADATA_CHECKS_ONLY'
    except BaseException as error:
        r.update(gate='FAIL_OFT_CPU_PREPARATION', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        (out / 'CPU_RECEIPT.json').write_text(json.dumps(r, indent=2))
    print(r['gate'], flush=True)


if __name__ == '__main__':
    main()
