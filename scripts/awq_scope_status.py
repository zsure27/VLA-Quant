"""Read-only finite monitor for a specified AWQ scope shard."""
import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--watch',action='store_true');a=p.parse_args()
for _ in range(240 if a.watch else 1):
    result=dict(directory=str(a.directory),batch_exit=None,cases={})
    exitfile=a.directory/'exit-code.txt'
    if exitfile.exists():result['batch_exit']=exitfile.read_text().strip()
    for case in a.directory.glob('*/command.txt'):
        d=case.parent;logs=list(d.glob('EVAL-*.txt'))
        text=logs[0].read_text(errors='replace') if len(logs)==1 else ''
        flags=[line[9:] for line in text.splitlines() if line.startswith('Success: ')]
        result['cases'][d.name]=dict(completed=len(flags),success=flags.count('True'),
            episode_errors=text.count('Episode error:'),exit=(d/'exit-code.txt').read_text().strip() if (d/'exit-code.txt').exists() else None)
    print(json.dumps(result),flush=True)
    if result['batch_exit'] is not None:break
    if a.watch:time.sleep(45)
