"""Create a compact summary from large per-sample P2 probe outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def average(rows, *keys):
    values=[]
    for row in rows:
        value=row
        for key in keys: value=value[key]
        values.append(float(value))
    return sum(values)/len(values)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    runs=[]
    for directory in sorted(args.session.iterdir()):
        if not directory.is_dir() or not (directory/"metrics.json").is_file(): continue
        rows=json.loads((directory/"metrics.json").read_text())
        scope=json.loads((directory/"scope.json").read_text())
        lora=scope.get("low_rank_residual",{})
        scale=scope.get("fixed_code_scale_peft",{})
        records=lora or scale
        e2e=scope.get("e2e_action_distillation") or {}
        runs.append({"name":directory.name,"samples":len(rows),
            "exact_12l_backbone":bool(scope.get("exact_attention_visual_stage",False)),
            "normalized_action_mse":average(rows,"normalized_action","mse"),
            "raw_action_mse":average(rows,"raw_action","mse"),
            "raw_gripper_disagreement":average(rows,"raw_gripper_disagreement"),
            "adapter_parameters":scope.get("residual_adapter_parameters",0) or scope.get("scale_peft_parameters",0),
            "training_steps":e2e.get("steps",0) or scope.get("residual_training_steps",0) or scope.get("scale_peft_training_steps",0),
            "training_objective":e2e.get("objective","layer_response_mse" if records else None),
            "target_count":len(records),
            "e2e_initial_train_mse":e2e.get("initial_train_mse"),
            "e2e_final_train_mse":e2e.get("final_train_mse"),
            "training_initial_mse_mean":sum((v.get("training_initial_mse") or 0) for v in records.values())/max(1,len(records)),
            "training_final_mse_mean":sum((v.get("training_final_mse") or 0) for v in records.values())/max(1,len(records))})
    baseline=next((run for run in runs if run["name"]=="exact12l-baseline-offline32"),None)
    for run in runs:
        if baseline:
            run["normalized_mse_delta_vs_baseline"]=run["normalized_action_mse"]-baseline["normalized_action_mse"]
            run["raw_mse_delta_vs_baseline"]=run["raw_action_mse"]-baseline["raw_action_mse"]
    improved=[run["name"] for run in runs if baseline and run["exact_12l_backbone"] and run["name"]!=baseline["name"] and
              run["normalized_action_mse"]<baseline["normalized_action_mse"] and
              run["raw_action_mse"]<baseline["raw_action_mse"] and
              run["raw_gripper_disagreement"]<=baseline["raw_gripper_disagreement"]]
    payload={"schema_version":"1.0","backbone":"awq-w2a16-12l-mixed-spatial-v1",
             "target_blocks":[18,19],"evaluation":"32 historical/development frames; offline agreement only",
             "runs":runs,"offline_gate_passed":bool(improved),"improved_runs":improved,
             "next_gate":"closed-loop development evaluation only if a shared method has positive offline signal"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({"runs":len(runs),"offline_gate_passed":bool(improved),"improved_runs":improved},sort_keys=True))


if __name__=="__main__": main()
