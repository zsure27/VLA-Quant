"""Stream-verify pi0.5 assets and produce blocked CPU-only inventory/templates.

No model load, inference, conversion, calibration, GPU registration or rollout.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

OPENPI = '215abfb217dbac7d5f1273282331b9b1866c0479'
LIBERO = '8f1084e3132a39270c3a13ebe37270a43ece2a01'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {'gate': 'IN_PROGRESS_CPU_ONLY', 'gpu_run_admitted': False,
              'observed_at': datetime.now(timezone.utc).isoformat(), 'sources': {}}
    try:
        for name, expected in [('openpi', OPENPI), ('LIBERO', LIBERO)]:
            p = root / 'sources' / name
            commit = subprocess.check_output(['git', '-C', str(p), 'rev-parse', 'HEAD'], text=True).strip()
            assert commit == expected
            assert not subprocess.check_output(['git', '-C', str(p), 'status', '--porcelain'], text=True)
            report['sources'][name] = commit
        native = root / 'models/pi05_libero'
        receipt = native / 'DOWNLOAD_VERIFICATION.json'
        original = json.loads(receipt.read_text())
        total = 0
        for item in original['files']:
            rel = Path(item['path'])
            assert not rel.is_absolute() and '..' not in rel.parts
            file = native / rel
            assert file.is_file() and not file.is_symlink()
            assert file.stat().st_size == item['bytes'] and sha(file) == item['sha256'], str(file)
            total += item['bytes']
        assert total == original['bytes'] and len(original['files']) == original['objects']
        report['native_checkpoint'] = {'files_verified': len(original['files']), 'bytes': total,
                                       'receipt_sha256': sha(receipt), 'source': original['source']}
        print('PASS_NATIVE_CHECKPOINT_SHA', total, flush=True)
        converted = root / 'models/pi05_libero_pytorch_bridge_v2/model.safetensors'
        expected = (root / 'checks/pi05-pytorch-bridge-v2.sha256').read_text().split()[0]
        assert sha(converted) == expected
        with converted.open('rb') as stream:
            n = int.from_bytes(stream.read(8), 'little')
            assert 0 < n < 64 * 1024 * 1024
            header = json.loads(stream.read(n))
        tensors = {k: v for k, v in header.items() if k != '__metadata__'}
        candidates = {k: v for k, v in tensors.items() if k.endswith('.weight') and len(v['shape']) == 2
                      and '.layers.' in k and any(x in k for x in ('.q_proj.', '.k_proj.', '.v_proj.',
                      '.o_proj.', '.gate_proj.', '.up_proj.', '.down_proj.'))}
        assert candidates
        report['converted_checkpoint'] = {'sha256': expected, 'bytes': converted.stat().st_size,
                                          'tensor_count': len(tensors), 'candidate_linear_count': len(candidates),
                                          'full_model_numerical_admission': 'BLOCKED_PREVIOUS_BF16_BRIDGE_FAILURE'}
        (out / 'converted-candidate-layer-inventory.json').write_text(json.dumps({
            'scope': 'inventory only; not a validated method target selection', 'layers': candidates}, indent=2))
        print('PASS_CONVERTED_SHA_AND_HEADER', len(tensors), len(candidates), flush=True)
        patch_receipt = root / 'checks/20261109-openpi-prescribed-transformers-patch.json'
        patch = json.loads(patch_receipt.read_text())
        installed = Path(next(iter(importlib.util.find_spec('transformers').submodule_search_locations)))
        for row in patch['files']:
            assert sha(installed / row['path']) == row['source_sha256'], row['path']
        report['prescribed_transformers_patch'] = {'files_verified': len(patch['files']),
                                                   'receipt_sha256': sha(patch_receipt)}
        bridges = []
        for folder in ['pi05-same-input-torch-v4', 'pi05-same-input-torch-v5']:
            path = root / 'checks' / folder / 'result.json'
            body = json.loads(path.read_text())
            assert body['gate'] == 'FAIL_PI05_SAME_INPUT_TORCH_BRIDGE', body['gate']
            bridges.append({'path': str(path.relative_to(root)), 'sha256': sha(path),
                            **{k: body[k] for k in ('gate', 'rmse', 'max_abs', 'query_sha256')}})
        report['preserved_bridge_failures'] = bridges
        report['bridge_thresholds_unchanged'] = {'rmse_max': .002, 'max_abs_max': .02}
        packages = {}
        for name in ['torch', 'jax', 'jaxlib', 'transformers', 'numpy', 'mujoco', 'robosuite', 'gym-aloha']:
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        check = subprocess.run(['uv', 'pip', 'check', '--python', str(root / 'envs/pi05/bin/python')],
                               text=True, capture_output=True)
        report['dependency_check'] = {'exit_code': check.returncode, 'output': check.stdout + check.stderr}
        report['package_versions'] = packages
        report['environment'] = {'hostname': subprocess.check_output(['hostname'], text=True).strip(),
                                 'gpu_inventory': subprocess.run(['nvidia-smi', '-L'], text=True, capture_output=True).stdout.strip(),
                                 'memory_max': Path('/sys/fs/cgroup/memory.max').read_text().strip(),
                                 'disk_free_bytes': shutil.disk_usage(root).free,
                                 'new_method_min_free_bytes': 12 * 1024**3}
        catalog = json.loads((root / 'code/configs/infra/runtime_commonality_v6/method_catalog.json').read_text())
        for suite in ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']:
            for index, item in enumerate(catalog['methods']):
                config = {'model': 'pi05', 'suite': suite, 'tasks': list(range(10)),
                          'initial_states': [5, 15, 25, 35, 45], 'method': item['method'],
                          'precision': item['precision'], 'executable': False, 'gpu_run_admitted': False,
                          'native_execute_horizon': 5, 'flow_action_horizon': 10, 'action_internal_dim': 32,
                          'native_checkpoint_receipt_sha256': sha(receipt), 'converted_checkpoint_sha256': expected,
                          'input_and_noise_pairing_required': True,
                          'blockers': ['full-model BF16 bridge failed; thresholds unchanged',
                                       'faithful pi05 method binding and calibration not admitted'],
                          'do_not_reuse_other_models_profiles': True, 'purpose': 'trajectory measurement only'}
                (out / f'{suite}-{index:02d}-{item["method"].lower()}-{item["precision"].lower()}.json').write_text(json.dumps(config, indent=2))
        report['blocked_method_templates'] = 4 * len(catalog['methods'])
        report['gate'] = 'PASS_PI05_CPU_ASSET_AND_METADATA_CHECKS_ONLY'
    except BaseException as error:
        report.update(gate='FAIL_PI05_CPU_PREPARATION', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        (out / 'CPU_RECEIPT.json').write_text(json.dumps(report, indent=2))
    print(report['gate'], flush=True)


if __name__ == '__main__':
    main()
