"""Separate sequential BF16 teacher pass on saved student query observations.

Not a rollout, not online gating, not a training command. Raw observations remain
private. Requires the same independently verified admission as the collector.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture-root",type=Path,required=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--admission",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise ValueError("Fresh teacher output required")
    gate=json.loads(args.admission.read_text(encoding="utf-8"))
    if gate.get("gate")!="PASS_TRAJECTORY_TEACHER_SAME_INPUT_V1" or not all(gate.get(k) is True for k in ("raw_proprio_once", "same_processed_images", "correct_suite_statistics", "finite", "frozen", "reload")):
        raise ValueError("Teacher actual-model admission missing")
    from qvla.baseline_contract import checkpoint_identity
    checkpoint_sha=checkpoint_identity(args.checkpoint)["sha256"]
    if gate.get("checkpoint_sha256") != checkpoint_sha: raise ValueError("Teacher checkpoint mismatch")
    from qvla.runtime_contract import assert_oft_runtime
    assert_oft_runtime()
    import torch
    from experiments.robot.libero import run_libero_eval as L
    from qvla.on_policy_capture import observation_hash
    from qvla.reproducibility import seed_all
    from qvla.trajectory_diagnostics_v1 import same_observation_chunk, finite
    episodes=sorted(args.capture_root.glob("task-*-reset-*/episode.json"))
    if not episodes: raise ValueError("No captured episodes")
    metas=[json.loads(p.read_text(encoding="utf-8")) for p in episodes]
    suites=set(m["suite"] for m in metas)
    if len(suites)!=1 or any(m["checkpoint_sha256"]!=checkpoint_sha or m["data_kind"]!="measured" for m in metas):
        raise ValueError("Teacher pass requires one suite and matching measured checkpoint")
    cfg=L.GenerateConfig(pretrained_checkpoint=args.checkpoint,task_suite_name=next(iter(suites)),
                         use_l1_regression=True,use_diffusion=False,use_film=False,
                         num_images_in_input=2,use_proprio=True,center_crop=True,seed=7)
    started=time.perf_counter()
    seed_all(7,tensorflow=True)
    model,head,proprio,noisy,processor=L.initialize_model(cfg)
    model.language_model.config.use_cache=False
    model.vision_backbone.set_num_images_in_input(2)
    for component in (model,head,proprio):
        component.requires_grad_(False); component.eval()
    args.output.mkdir(parents=True)
    count=0
    with (args.output/"same-input-metrics.jsonl").open("x",encoding="utf-8") as stream:
        for path,meta in zip(episodes,metas):
            seed_all(meta["model_seed"],tensorflow=True)
            events=[json.loads(s) for s in (path.parent/"steps.jsonl").read_text(encoding="utf-8").splitlines()]
            queries={e["query_index"]:e for e in events if e["kind"]=="policy_query"}
            for query in sorted(path.parent.glob("query-*.npz")):
                index=int(query.stem.split("-")[-1])
                with np.load(query,allow_pickle=False) as data:
                    obs={"full_image":data["image"].copy(),"wrist_image":data["wrist_image"].copy(),"state":data["state"].copy()}
                    instruction=str(data["instruction"].item())
                    student=data["raw_policy_chunk"].copy()
                digest=observation_hash(obs["full_image"],obs["wrist_image"],obs["state"],instruction)
                if digest!=queries[index]["input_sha256"]: raise ValueError("Saved input changed")
                with torch.inference_mode():
                    teacher=finite(L.get_action(cfg,model,obs,instruction,processor=processor,
                                   action_head=head,proprio_projector=proprio,noisy_action_projector=noisy,use_film=False))
                metrics=same_observation_chunk(student,teacher,digest,queries[index]["input_sha256"])
                student_cmd=np.asarray([L.process_action(x.copy(),cfg.model_family) for x in student])
                teacher_cmd=np.asarray([L.process_action(x.copy(),cfg.model_family) for x in teacher])
                metrics["gripper_after_processing_disagreement"]=(student_cmd[:,-1]!=teacher_cmd[:,-1]).tolist()
                row=dict(suite=meta["suite"],task_id=meta["task_id"],reset_index=meta["reset_index"],
                         candidate=meta["model_id"],query_index=index,input_sha256=digest,metrics=metrics,
                         executed_chunk_positions=[e["chunk_position"] for e in events
                             if e["kind"]=="control_step" and e["query_index"]==index])
                stream.write(json.dumps(row,allow_nan=False)+"\n"); stream.flush()
                np.savez_compressed(args.output/(path.parent.name+"-"+query.name),teacher_chunk=teacher,input_sha256=np.asarray(digest))
                count+=1
    (args.output/"receipt.json").write_text(json.dumps({"status":"completed","queries":count,
        "teacher_load_and_label_wall_s":time.perf_counter()-started,"checkpoint_sha256":checkpoint_sha,
        "admission_sha256":hashlib.sha256(args.admission.read_bytes()).hexdigest(),
        "note":"No teacher actions executed. Not a deployment-latency measurement."},indent=2),encoding="utf-8")


if __name__=="__main__": main()
