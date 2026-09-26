#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${VLA_EXPERIMENT_ROOT:-/root/autodl-tmp/qvla-repro}
REPO=${VLA_REPO:-/root/autodl-tmp/VLA-Quant-p2c-20260925}
SESSION=${VLA_SESSION:-20260926-107-p2-5-on-policy-alignment}
OUT=$ROOT/backups/experiments/p2-shared-peft/$SESSION
LORA_STATE=${VLA_LORA_STATE:?VLA_LORA_STATE must name the exact data80 Recovery-LoRA checkpoint}
BASE=$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w2.pt
W4=$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w4.pt
G64=$ROOT/artifacts/awq-primary-group-20260913-213127-3342/profiles/w2-g64.pt
OFT_ROOT=${VLA_OFT_ROOT:-$ROOT/overlays/awq-p0-stage-20260923/oft}
test ! -e "$OUT"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || exit 3
mkdir -p "$OUT/student-rollout"

cat > "$OUT/preregistered-protocol.json" <<'JSON'
{
  "schema_version": "1.0",
  "question": "Can compact rank8 Recovery-LoRA remain close to BF16 on states induced by its own actions?",
  "status": "PREREGISTERED_BEFORE_EXECUTION",
  "stage": "P2.5_closed_loop_alignment_diagnosis",
  "backbone": "awq-w2a16-12l-mixed-spatial-v1",
  "target_blocks": [18, 19],
  "rank": 8,
  "student_rollouts": {"suite": "libero_spatial", "initial_state_offset": 0, "trials_per_task": 1, "episodes": 10},
  "same_observation_contract": "capture observation passed to student get_action, then query frozen BF16 on the serialized identical observation",
  "action_chunk_contract": "all 8 actions execute open-loop before the next policy query",
  "future_success_used_as_input": false,
  "large_divergence_threshold": null,
  "threshold_reason": "router_dev distribution threshold not yet frozen; record continuous disagreements without post-hoc classification",
  "classification": "P2.5 mechanism diagnosis, historical development resets; not an unbiased final benchmark"
}
JSON

source /root/miniconda3/bin/activate /root/miniconda3/envs/qvla-oft
export PYTHONPATH="$REPO:$OFT_ROOT:$ROOT/src/LIBERO"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 WANDB_MODE=disabled MUJOCO_GL=egl
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4 TOKENIZERS_PARALLELISM=false

python "$REPO/qvla/run_eval_official_quant.py" \
  --method awq --weight-bits 2 --activation-bits 16 \
  --pretrained_checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial" \
  --profile "$BASE" --awq-primary-group64-profile "$G64" --awq-w4-profile "$W4" \
  --official-root "$ROOT/src/official-quantization" --task_suite_name libero_spatial \
  --num_trials_per_task 1 --initial-state-offset 0 --libero_root "$ROOT/src/LIBERO" \
  --seed 0 --env-seed 0 --seed-protocol paired --trace-actions --trace-observations --awq-scope all \
  --awq-candidate w2-attention-primary-g64-stage-w4 \
  --awq-w4-layers 8,9,10,11,12,13,14,15,20,21,22,23 \
  --awq-recovery-lora-state "$LORA_STATE" --local_log_dir "$OUT/student-rollout" \
  2>&1 | tee "$OUT/student-rollout-console.log"
printf '0\n' > "$OUT/student-rollout-exit-code.txt"

python "$REPO/qvla/query_same_observation_teacher.py" \
  --checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial" \
  --student-run "$OUT/student-rollout" --output "$OUT/bf16-same-observation" --seed 0 \
  2>&1 | tee "$OUT/teacher-query-console.log"
printf '0\n' > "$OUT/teacher-query-exit-code.txt"
(cd "$OUT" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS.txt)
printf '{"status":"COMPLETE","stage":"P2.5","future_success_used_as_input":false}\n' > "$OUT/complete.json"

