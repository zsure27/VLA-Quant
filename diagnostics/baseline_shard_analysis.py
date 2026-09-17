"""Analyze paired official-index shards from unwrapped original EVAL logs."""
import json,re
from pathlib import Path

def parse_episodes(text):
 rows=[];pending=None
 for line in text.splitlines():
  if line.startswith("EPISODE_MANIFEST "):
   if pending is not None:raise ValueError("new manifest before episode outcome")
   pending=dict(json.loads(line.split(" ",1)[1]));pending["episode_errors"]=[]
  elif "Episode error:" in line and pending is not None:
   pending["episode_errors"].append(line)
  elif line.startswith("Success: "):
   if pending is None:raise ValueError("outcome without manifest")
   flag=line.split(": ",1)[1]
   if flag not in ("True","False"):raise ValueError("invalid success flag")
   pending["success"]=flag=="True";rows.append(pending);pending=None
 ids=[(r["task_id"],r["init_state_index"]) for r in rows]
 if len(ids)!=len(set(ids)):raise ValueError("duplicate official initial state")
 return rows

def analyze(directory):
 directory=Path(directory);cases={}
 for case in ("bf16","w4"):
  d=directory/case;logs=list(d.glob("EVAL-*.txt"))
  if len(logs)!=1:raise ValueError("one original EVAL log required per variant")
  command=(d/"command.txt").read_text()
  count=int(re.search(r"--num_trials_per_task\s+(\d+)",command).group(1))
  offset=int(re.search(r"--initial-state-offset\s+(\d+)",command).group(1))
  rows=parse_episodes(logs[0].read_text());expected={(t,i) for t in range(10) for i in range(offset,offset+count)}
  if {(r["task_id"],r["init_state_index"]) for r in rows}!=expected:raise ValueError("incomplete or wrong official-index coverage")
  if (d/"exit-code.txt").read_text().strip()!="0":raise ValueError("variant did not finish exit0")
  cases[case]=rows
 if len(cases["bf16"])!=len(cases["w4"]):raise ValueError("paired count mismatch")
 pairs=[]
 for b,w in zip(cases["bf16"],cases["w4"]):
  if {k:v for k,v in b.items() if k not in ("success","episode_errors")}!={k:v for k,v in w.items() if k not in ("success","episode_errors")}:raise ValueError("paired manifest mismatch")
  pairs.append(dict(task_id=b["task_id"],init_state_index=b["init_state_index"],model_seed=b["model_seed"],env_seed=b["env_seed"],init_state_sha256=b["init_state_sha256"],bf16_success=b["success"],w4_success=w["success"],bf16_episode_errors=b["episode_errors"],w4_episode_errors=w["episode_errors"]))
 return dict(directory=str(directory),paired_episodes=pairs,summary={c:dict(episodes=len(rs),successes=sum(r["success"] for r in rs),episode_errors=sum(len(r["episode_errors"]) for r in rs)) for c,rs in cases.items()})

if __name__=="__main__":
 import argparse
 p=argparse.ArgumentParser();p.add_argument("directory",type=Path);a=p.parse_args();r=analyze(a.directory)
 (a.directory/"paired-results.json").write_text(json.dumps(r,indent=2)+"\n")
 print(json.dumps(r["summary"]))
