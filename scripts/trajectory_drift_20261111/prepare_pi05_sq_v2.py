"""Sequential single-model capture and profile validation before SQ rollouts."""
import argparse
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--parent-round',type=Path,required=True);p.add_argument('--round-dir',type=Path,required=True)
a=p.parse_args();scripts=Path(__file__).parent
subprocess.run([sys.executable,str(scripts/'capture_pi05_calibration_v2.py'),'--root',str(a.root),
                '--parent-round',str(a.parent_round),'--output',str(a.round_dir/'capture')],check=True)
subprocess.run([sys.executable,str(scripts/'build_pi05_smoothquant_v2.py'),'--root',str(a.root),
                '--capture',str(a.round_dir/'capture'),'--output',str(a.round_dir/'profile')],check=True)
