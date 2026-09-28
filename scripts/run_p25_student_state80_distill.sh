#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${VLA_EXPERIMENT_ROOT:-/root/autodl-tmp/qvla-repro}
REPO=${VLA_REPO:-/root/autodl-tmp/VLA-Quant-p2c-20260925}
SESSION=${VLA_SESSION:-20260929-107-p25-student-state80-distill}
OUT_ROOT="$ROOT/backups/experiments/p2-shared-peft/$SESSION"
CALIBRATION=${VLA_PEFT_CALIBRATION:-$ROOT/backups/experiments/p1-data-contract/20260926-107-p1-data-contract/peft-train-calibration80}
OUT="$OUT_ROOT/exact12l-response-svd-r8-smoothl1-studentstate80-e2e1000"
test ! -e "$OUT"
mkdir -p "$OUT_ROOT"
SOURCE_MATCHES=("$ROOT"/eval/p25-paired-headroom-0-9-*)
(( ${#SOURCE_MATCHES[@]} == 1 )) || exit 3
SOURCE=${SOURCE_MATCHES[0]}
SAMPLES=$OUT_ROOT/student-state80
test -s "$SOURCE/complete.json"
test -s "$SOURCE/C1-12L-lora/on-policy-events.jsonl"
test ! -e "$SAMPLES"

export PYTHONPATH="$REPO/diagnostics:$REPO:$ROOT/src/QVLA/openvla-oft:$ROOT/src/LIBERO"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=7 WANDB_MODE=disabled
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4

/root/miniconda3/envs/qvla-oft/bin/python "$REPO/scripts/prepare_p25_student_state80.py" \
  --paired-shard "$SOURCE" --student-run "$SOURCE/C1-12L-lora" --output "$SAMPLES"

args=(
  --checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial"
  --samples-dir "$ROOT/artifacts/awq-validation-20260913-132827-1162/samples"
  --official-root "$ROOT/src/official-quantization"
  --targets-file "$REPO/configs/qvla-connected-422.txt"
  --num-samples 32 --offset 0 --seed 7 --attention-layers 7,15,23,31
  --teacher-dir "$ROOT/artifacts/awq-validation-20260913-132827-1162/teacher"
  --mode awq --weight-bits 2 --activation-bits 16 --weight-scope all
  --awq-profile "$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w2.pt"
  --awq-disable-clip attention
  --awq-w4-layers 8,9,10,11,12,13,14,15,20,21,22,23
  --awq-w4-profile "$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w4.pt"
  --awq-vision-bits 2 --awq-vision-branch all
  --awq-primary-group64-profile "$ROOT/artifacts/awq-primary-group-20260913-213127-3342/profiles/w2-g64.pt"
  --awq-exact-attention-visual-stage
  --awq-residual-layers 18,19 --awq-residual-rank 8
  --awq-residual-calibration-dir "$CALIBRATION" --awq-residual-calibration-count 80
  --awq-residual-token-scope action --awq-residual-response-svd
  --awq-e2e-distill-steps 1000 --awq-e2e-distill-learning-rate 0.0001
  --awq-e2e-distill-loss smooth-l1 --awq-e2e-distill-smooth-l1-beta 0.1
  --awq-e2e-distill-samples-dir "$SAMPLES" --awq-e2e-distill-samples-count 80
  --output "$OUT"
)

cat > "$OUT_ROOT/preregistered-protocol.json" <<'JSON'
{
  "schema_version": "1.0",
  "question": "Does BF16 relabeling on student-visited training resets improve closed-loop-aligned Recovery-LoRA without changing rank, initialization, objective, optimizer or steps?",
  "status": "PREREGISTERED_BEFORE_EXECUTION",
  "backbone": "awq-w2a16-12l-mixed-spatial-v1",
  "target_blocks": [18, 19],
  "interpretation_correction_20260929": "Response-SVD calibration remains the original peft_train 80; only the 80 action-distillation observations change. Reset indices 0-4 are excluded from later rollout evaluation.",
  "changed_factors": {"action_distillation_sample_distribution": "80 peft_train demonstration states -> 80 LoRA student-visited training-reset states, BF16 same-observation relabel"},
  "fixed_controls": {"rank": 8, "initialization": "Response-SVD from peft_train 80", "loss": "Smooth L1(beta=0.1), uniform 8x7", "training_samples": 80, "steps": 1000, "learning_rate": 0.0001, "offline_frames": 32},
  "offline_gate": {"median_paired_mse_delta_lt": 0, "improved_frames_gte": 20, "mean_mse_not_worse": true, "gripper_disagreement_not_worse": true},
  "classification": "P2.5 student-state distribution intervention; evaluate only reset 5-49 for leakage-free closed-loop evidence"
}
JSON

printf '%q ' /root/miniconda3/envs/qvla-oft/bin/python -u "$REPO/diagnostics/probe.py" "${args[@]}" > "$OUT_ROOT/command.txt"
printf '\n' >> "$OUT_ROOT/command.txt"
/root/miniconda3/envs/qvla-oft/bin/python -u "$REPO/diagnostics/probe.py" "${args[@]}" 2>&1 | tee "$OUT_ROOT/console.log"
(cd "$OUT" && find . -type f -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS.txt)
printf '{"status":"COMPLETE","objective":"normalized_action_smooth-l1_8x7","training_distribution":"student_visited_train_reset_0_3","beta":0.1,"training_samples":80,"training_steps":1000}\n' > "$OUT_ROOT/complete.json"
(cd "$OUT_ROOT" && find . -type f ! -name SESSION_SHA256SUMS.txt -print0 | sort -z | xargs -0 -r sha256sum > SESSION_SHA256SUMS.txt)
