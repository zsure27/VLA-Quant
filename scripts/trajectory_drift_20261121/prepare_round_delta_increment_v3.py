"""Snapshot this stopped round, sources and calibration32; keep parameters server-side.

No network or deletion. The existing archive tool verifies every tar member.
Historical raw archives are referenced explicitly rather than copied again.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--parent-archive-proof', type=Path, required=True)
    p.add_argument('--transfer-budget-gib', type=float, default=4)
    args = p.parse_args()
    r, root = args.root.resolve(), args.control.resolve()
    assert root.parent == r / 'control'
    parent_proof = json.loads(args.parent_archive_proof.read_text())
    assert parent_proof['gate'] == 'PASS_CURRENT_INCREMENT_ARCHIVE_MEMBER_SHA'
    parent_archive = parent_proof['archive']; archive_path = Path(parent_archive['path']).resolve()
    assert archive_path.is_relative_to(r / 'control') and not archive_path.is_symlink()
    assert archive_path.stat().st_size == parent_archive['bytes'] and sha(archive_path) == parent_archive['sha256']
    with tarfile.open(archive_path, 'r:gz') as tar:
        parent_manifest = json.load(tar.extractfile('ARCHIVE_MEMBERS.json'))
    parent_members = {row['path']: row for row in parent_manifest['members']}
    assert len(parent_members) == len(parent_manifest['members'])
    reused = []
    remaining_archive_budget = int(args.transfer_budget_gib * 1024**3 / 3) - parent_archive['bytes']
    assert remaining_archive_budget > 0

    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    interrupted = []
    for stage in (root / 'stages').iterdir():
        if not (stage / 'status.json').exists():
            continue
        state = json.loads((stage / 'status.json').read_text())
        if state.get('state') in ('RUNNING', 'PREFLIGHT'):
            receipt = json.loads((stage / 'launch-receipt.json').read_text())
            cmdline = Path('/proc', str(receipt['supervisor_pid']), 'cmdline')
            assert not cmdline.exists() or str(stage).encode() not in cmdline.read_bytes()
            interrupted.append(dict(stage=str(stage), old_status_preserved=True, excluded_from_complete_stage_counts=True))
    area = root / '.closure/current-increment-v3'
    area.mkdir(parents=True, exist_ok=False)
    excluded, inputs, sources = {}, {}, {}

    def copy(source, dest):
        assert source.is_file() and not source.is_symlink() and source.is_relative_to(r)
        expected = sha(source)
        relative = dest.relative_to(area).as_posix()
        prior = parent_members.get(relative)
        if prior and prior['sha256'] == expected and prior['bytes'] == source.stat().st_size:
            reused.append(dict(restored_path=relative, parent_member=relative, bytes=prior['bytes'], sha256=expected))
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        assert sha(dest) == expected

    def retain(source):
        excluded[str(source)] = dict(path=str(source), bytes=source.stat().st_size, sha256=sha(source), reason='User keeps parameters on persistent server disk; not uploaded')

    for source in sorted(root.rglob('*')):
        rel = source.relative_to(root)
        if '.closure' in rel.parts or '__pycache__' in rel.parts:
            continue
        assert not source.is_symlink()
        if not source.is_file():
            continue
        if source.suffix in ('.pt', '.pth', '.safetensors', '.bin'):
            retain(source)
        else:
            copy(source, area / 'current-control' / rel)
    for file in (root / 'stages').glob('*/plan.json'):
        plan = json.loads(file.read_text())
        sources.update(plan['source_sha256'])
        for row in plan['inputs']:
            inputs[row['path']] = row
        for row in plan['config'].get('quantization', {}).get('profiles', []):
            source = Path(row['path'])
            assert sha(source) == row['sha256']
            retain(source)
    current_code = Path(json.loads((root / 'CODE_ROOT_LINEAGE_V1.json').read_text())['current_code'])
    assert current_code.is_relative_to(r)
    for source in current_code.rglob('*'):
        rel = source.relative_to(current_code)
        if source.is_file() and '.git' not in rel.parts and '__pycache__' not in rel.parts and source.suffix in ('.py', '.json'):
            assert source.stat().st_size < 2 * 1024**2
            expected = sha(source)
            if str(source) in sources:
                assert sources[str(source)] == expected
            sources[str(source)] = expected
    for name, expected in sources.items():
        source = Path(name)
        assert sha(source) == expected
        copy(source, area / 'scientific-source' / source.relative_to(r))
    capture = r / 'rounds/20261111-084-pi05-smoothquant-a8-v2/capture'
    # statistics.pt is calibration statistics/input indexing, not model weights.
    import torch
    cache = torch.load(capture / 'statistics.pt', map_location='cpu', weights_only=True)
    assert len(cache['samples']) == 32
    for source in (capture / 'statistics.pt', capture / 'result.json'):
        copy(source, area / 'calibration32' / source.name)
    for row in cache['samples']:
        source = Path(row['path'])
        assert row['role'] == 'calibration' and row['reset_index'] == 0 and sha(source) == row['sha256']
        copy(source, area / 'calibration32/raw' / source.relative_to(r))
    readiness = r / 'preparations/admission-first-20261119-v1/readiness/readiness.json'
    copy(readiness, area / 'dependency-metadata/readiness.json')
    for directory in (r/'models/pi05_libero', r/'models/pi05_libero_pytorch_bridge_v2'):
        for source in directory.rglob('*.json'):
            assert source.stat().st_size < 2 * 1024**2
            copy(source, area / 'dependency-metadata' / source.relative_to(r))
    import importlib.metadata
    versions = {}
    for package in ('torch','numpy','transformers','jax','jaxlib','mujoco','scipy','matplotlib'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    (area/'dependency-metadata').mkdir(parents=True, exist_ok=True)
    (area/'dependency-metadata/runtime-versions.json').write_text(json.dumps(versions,indent=2))
    parent = json.loads((root / 'PARENT_RECOVERY_INDEX.json').read_text())
    parent_root = Path(parent['parent_control'])
    for file in ('CLONE_REUSE_VERIFICATION_V1.json', 'CODE_ROOT_LINEAGE_V1.json', 'task-manifest.json', 'shutdown-receipt.json'):
        if (parent_root / file).exists():
            copy(parent_root / file, area / 'parent-provenance' / file)
    (area / 'REUSED_PARENT_MEMBERS_V1.json').write_text(json.dumps(dict(parent_archive=parent_archive, parent_cloud_verification_not_assumed=True, reused=reused), indent=2))
    (area / 'PARAMETERS_RETAINED_ON_SERVER.json').write_text(json.dumps(dict(files=list(excluded.values()), cloud_parameter_gap=True), indent=2))
    (area / 'DEPENDENCY_AND_HISTORY_GAPS.json').write_text(json.dumps(dict(parent=parent, external_inputs=list(inputs.values()), historical_raw_not_duplicated=True, historical_raw_not_claimed_uploaded=True, model_checkpoint_server_only=True, interrupted_stages=interrupted), indent=2))
    (area / 'status.json').write_text(json.dumps(dict(state='STOPPED_CURRENT_INCREMENT_SNAPSHOT', parameters_not_uploaded=True)))
    auth = json.loads((root / 'authorization.json').read_text())
    entry = Path(auth['control_code']) / 'scripts/infra_archive_run.py'
    subprocess.run([sys.executable, str(entry), '--run-dir', str(area), '--max-upload-gib', str(remaining_archive_budget / 1024**3), '--prepare-only'], check=True)
    plan = json.loads((area / '.closure/upload-plan.json').read_text())
    assert (parent_archive['bytes'] + plan['files'][0]['bytes']) * 3 <= args.transfer_budget_gib * 1024**3
    proof = dict(gate='PASS_CURRENT_DELTA_ARCHIVE_MEMBER_SHA', archive=plan['files'][0], minimum_receive_upload_readback_bytes=plan['files'][0]['bytes']*3, total_parent_plus_delta_three_leg_bytes=(parent_archive['bytes']+plan['files'][0]['bytes'])*3, parent_archive_required=parent_archive, reused_parent_members=len(reused), parameters_retained=len(excluded), historical_parent_cloud_gap=True, private_cloud_verified=False)
    (root / 'FINAL_DELTA_ARCHIVE_VERIFIED_V3.json').write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof))


if __name__ == '__main__':
    main()
