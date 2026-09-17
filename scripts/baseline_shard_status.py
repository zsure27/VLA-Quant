import argparse,json,re,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path("/root/autodl-tmp/qvla-repro/eval"));p.add_argument("--watch",action="store_true");a=p.parse_args()
for iteration in range(240 if a.watch else 1):
 out=[]
 for d in sorted(a.root.glob("awq-baseline-shard-*")):
  item={"directory":str(d),"batch_exit":(d/"exit-code.txt").read_text().strip() if (d/"exit-code.txt").exists() else None}
  for case in ("bf16","w4"):
   logs=list((d/case).glob("EVAL-*.txt"));text=logs[0].read_text() if logs else ""
   flags=re.findall(r"^Success: (True|False)$",text,re.M)
   item[case]={"completed":len(flags),"success":flags.count("True"),"episode_errors":text.count("Episode error:"),"exit":(d/case/"exit-code.txt").read_text().strip() if (d/case/"exit-code.txt").exists() else None}
  out.append(item)
 print(json.dumps(out),flush=True)
 if not a.watch or (out and all(r["batch_exit"] is not None for r in out)):break
 time.sleep(45)
