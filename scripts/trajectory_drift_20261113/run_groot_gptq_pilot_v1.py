"""One predeclared pilot: sequential artifact admission then trajectory, one model at a time."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--round-dir',type=Path,required=True);p.add_argument('--reference-map',type=Path,required=True);p.add_argument('--suite',required=True);p.add_argument('--artifact-root',type=Path,required=True);a=p.parse_args()
refs=json.loads(a.reference_map.read_text());reference=Path(refs[a.suite]);assert reference.name==a.suite
artifact=a.artifact_root/a.suite;repo=Path(__file__).resolve().parents[2]
commands=[[sys.executable,str(repo/'scripts/trajectory_drift_20261113/prepare_groot_gptq_v3.py'),'--root',str(a.root),'--round-dir',str(a.round_dir),'--reference-round',str(reference.parent),'--suite',a.suite,'--output',str(artifact)], [sys.executable,'-m','qvla.groot_gptq_trace_20261113_v1','--root',str(a.root),'--round-dir',str(a.round_dir),'--reference-round',str(reference.parent),'--suite',a.suite,'--profile',str(artifact/'profile.pt')]]
for command in commands:
 print(json.dumps(dict(actual_child_argv=command)),flush=True);subprocess.run(command,cwd=repo,check=True)
