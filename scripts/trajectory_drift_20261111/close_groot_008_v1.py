"""Prepare an incremental snapshot, then give the active controller a bounded relay window."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2]
subprocess.run([sys.executable,str(root/'scripts/infra_archive_run.py'),'--run-dir',str(a.run_dir),'--prepare-only'],check=True)
(a.run_dir/'relay-ready.json').write_text(json.dumps({'state':'WAITING_CONTROLLER_RELAY','maximum_seconds':900}))
start=time.monotonic()
while time.monotonic()-start<900:
    receipt=a.run_dir/'backup-receipt.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('remote_download_sha256_verified') is True:
        break
    time.sleep(2)
else:
    raise RuntimeError('Controller relay unavailable; server originals retained. Native shutdown must still be attempted.')
