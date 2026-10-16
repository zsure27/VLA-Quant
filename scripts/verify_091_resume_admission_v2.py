"""Wait only on the server for R0, then verify exact old/new observation identity."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('control', 'old-control', 'output'):
        p.add_argument('--' + k, type=Path, required=True)
    p.add_argument('--preflight-sha256', required=True)
    a = p.parse_args()
    if a.output.exists() or socket.gethostname() != 'autodl-container-dfbc4bb0ff-1e85dd94':
        raise ValueError('Existing admission or wrong instance')
    status = a.control / 'control/20261016-091-residual-resume-v2-R0/status.json'
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        if (a.control / 'STOP_NEW_GPU').exists():
            raise ValueError('Stop new GPU marker set')
        if status.exists():
            s = json.loads(status.read_text())
            if s['state'] == 'FAILED':
                raise ValueError('R0 failed')
            if s['state'] == 'AWAITING_GATE_REVIEW':
                break
        time.sleep(10)
    else:
        raise TimeoutError('R0 did not complete within admission budget')
    current = a.control / 'actual-observation-repeatability.json'
    prior = a.old_control / 'actual-observation-repeatability.json'
    new, old = json.loads(current.read_text()), json.loads(prior.read_text())
    if new['gate'] != 'PASS_CPU_PROTOCOL_OBSERVATION_BRIDGE':
        raise ValueError('Current R0 is not a pass')
    fields = ('plan', 'sources', 'normalization_key', 'within_protocol', 'cross_protocol', 'runs')
    if any(new[k] != old[k] for k in fields):
        raise ValueError('Actual no-model observation/preprocessor content differs across reboot')
    f = a.control / 'recovery-preflight.json'
    if sha(f) != a.preflight_sha256:
        raise ValueError('Preflight SHA differs from preregistration')
    receipt = json.loads(f.read_text())
    if receipt['gate'] != 'PASS_M0_47_PLUS_3_RECOVERY_PREFLIGHT':
        raise ValueError('No verified completed 47')
    for path, expected in receipt['frozen_files'].items():
        if sha(path) != expected:
            raise ValueError('Frozen source/material or old completed output changed')
    if shutil.disk_usage(a.control).free < 12 * 1024**3:
        raise ValueError('Need12GiB for GPU admission')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise ValueError('GPU occupied; do not duplicate')
    result = {'gate': 'PASS_091_REBOOT_AND_RECOVERY_CPU_ADMISSION',
              'new_R0_sha256': sha(current), 'old_R0_sha256': sha(prior),
              'actual_no_model_observations_equal_across_reboot': True,
              'preflight_sha256': a.preflight_sha256,
              'free_persistent_bytes': shutil.disk_usage(a.control).free,
              'actual_first_query_GPU_bridge': 'Pending; not represented as CPU success'}
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
