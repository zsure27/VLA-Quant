"""CPU-only creation of blocked infra templates; no SSH, CUDA or launches."""
from __future__ import annotations
import json
from pathlib import Path


def build(repo, destination):
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    task={"id":"multisuite-trajectory-deviation-v1", "name_zh":"四套件多任务轨迹偏移诊断",
          "started_at":"2026-11-06T15:04:07+08:00",
          "hypothesis":"检验量化是否改变平移分散度、旋转有向漂移，以及是否存在跨初态重复的可补偿窗口；允许无拐点。",
          "changed_variable":"首轮固定多suite OFT检查点、输入、8动作执行与配对初态，仅改变权重量化档；不训练。"}
    for model,bits in [("bf16",16),("awq-w4a16",4),("awq-w2a16",2)]:
        template="bf16_multisuite.example.json" if bits==16 else "awq_multisuite.example.json"
        cfg=json.loads((Path(repo)/"configs/infra"/template).read_text(encoding="utf-8"))
        cfg["task"]=task
        cfg["control"]["server_id"]="UNASSIGNED_USER_OPENED_SERVER_REQUIRED"
        for suite in cfg["evaluation"]["suites"]:
            suite["resets_per_task"]=1
        cfg["quantization"]["weight_bits"]=bits
        if bits!=16:
            for profile in cfg["quantization"]["profiles"]:
                profile["path"]=profile["path"].replace("w4","w"+str(bits))
        command=["{python}","-m","qvla.run_trajectory_diagnostic_v1","--backend",
                 "bf16" if bits==16 else "awq","--identity","{repo}/.local/trajectory-study/"+model+".identity.json",
                 "--admission","{repo}/.local/trajectory-study/"+model+".admission.json",
                 "--capture-root","{output}/capture","--"]
        if bits==16:
            command += ["--pretrained_checkpoint","{checkpoint}","--task_suite_name","{suite}",
                        "--num_trials_per_task","{count}","--initial_state_offset","{offset}",
                        "--local_log_dir","{output}","--seed","7","--env_seed","0","--seed_protocol","paired"]
        else:
            command += ["--method","awq","--weight-bits",str(bits),"--activation-bits","16",
                        "--pretrained_checkpoint","{checkpoint}","--official-root",cfg["quantization"]["official_root"],
                        "--task_suite_name","{suite}","--num_trials_per_task","{count}",
                        "--initial-state-offset","{offset}","--local_log_dir","{output}",
                        "--seed","7","--env-seed","0","--seed-protocol","paired"]
            for profile in cfg["quantization"]["profiles"]: command += ["--profile",profile["path"]]
            command += cfg["quantization"].get("recipe_argv",[])
        cfg["binding"]={"id":"oft-trajectory-observer-v1-pending-gpu-admission",
            "verified_suites":[r["name"] for r in cfg["evaluation"]["suites"]],
            "quantizers":[cfg["quantization"]["method"]],"strategies":["none"],
            "gate_files":[{"path":"{repo}/.local/trajectory-study/"+model+".admission.json",
                           "sha256":"0"*64,"expected_gate":"PASS_TRAJECTORY_CAPTURE_ACTUAL_MODEL_V1"}],
            "evaluation_argv":command}
        cfg["diagnostic_note"]="BLOCKED TEMPLATE: interface fields only, not GPU verification. Fill real checkpoint/profile/gate SHA, server and data-role audit; keep rounds balanced across all four suites."
        (destination/(model+".example.json")).write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        identity={"data_kind":"measured","model_id":model,"backend":"bf16" if bits==16 else "awq",
                  "weight_bits":bits,"activation_bits":16,"checkpoint":cfg["model"]["checkpoint"],
                  "checkpoint_sha256":"0"*64,"cohort_id":"REPLACE_AFTER_ENV_AND_PROFILE_AUDIT"}
        (destination/(model+".identity.example.json")).write_text(json.dumps(identity,indent=2)+"\n",encoding="utf-8")
    study={"status":"DESIGN_AND_CPU_ONLY_NO_SERVER", "task":task,
           "suites":{"libero_spatial":220,"libero_object":280,"libero_goal":300,"libero_10":520},
           "task_ids":list(range(10)),"development_resets":[5,6,7],
           "candidate_profiles":["bf16","awq-w4a16","awq-w2a16"],"formal_episode_budget":360,
           "same_input_teacher_queries_upper_bound":9960,
           "control_steps_upper_bound":118800,"settling_steps_excluded":10,
           "rounds":"reset5 across all 4 suites and 3 profiles, then reset6, then reset7; sequential model execution",
           "thresholds":"Calibrate on independent FP-FP repeatability and practical tolerance; hash and freeze before main analysis. Demo 20mm/5deg are NOT adopted defaults.",
           "stopping":"15% stop expansion, 10% close, 3% reserve; completed grid cells retained; no partial cohort represented as full coverage.",
           "extensions":"Teacher intervention and original OpenVLA are separately budgeted follow-ups; no automatic activation-A4, training, routing or 450 episodes."}
    (destination/"study.json").write_text(json.dumps(study,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return study


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(build(Path(__file__).resolve().parents[1],args.output),ensure_ascii=False))
