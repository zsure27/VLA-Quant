#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${VLA_EXPERIMENT_ROOT:-/root/autodl-tmp/qvla-repro}
REPO=${VLA_REPO:-/root/autodl-tmp/VLA-Quant-mainline-20260925}
SESSION=${VLA_SESSION:-20260925-107-p2-shared-peft}
OUT_ROOT="$ROOT/backups/experiments/p2-shared-peft/$SESSION"
CALIBRATION=${VLA_PEFT_CALIBRATION:-$ROOT/backups/experiments/p1-data-contract/20260925-107-p1-data-contract/peft-train-calibration16}
mkdir -p "$OUT_ROOT"

export PYTHONPATH="$REPO/diagnostics:$REPO:$ROOT/src/QVLA/openvla-oft:$ROOT/src/LIBERO"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=7 WANDB_MODE=disabled
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4

common=(
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
  --awq-residual-calibration-dir "$CALIBRATION" --awq-residual-calibration-count 16
  --awq-residual-token-scope action --awq-e2e-distill-steps 200
  --awq-e2e-distill-learning-rate 0.0001
)

run_probe() {
  local name=$1
  shift
  local out="$OUT_ROOT/$name"
  test ! -e "$out"
  printf '%q ' /root/miniconda3/envs/qvla-oft/bin/python -u "$REPO/diagnostics/probe.py" "${common[@]}" "$@" --output "$out" > "$OUT_ROOT/$name.command.txt"
  printf '\n' >> "$OUT_ROOT/$name.command.txt"
  /root/miniconda3/envs/qvla-oft/bin/python -u "$REPO/diagnostics/probe.py" "${common[@]}" "$@" --output "$out" 2>&1 | tee "$OUT_ROOT/$name.console.log"
  (cd "$out" && find . -type f -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS.txt)
}

run_probe exact12l-standard-zero-r8-e2e200-offline32
run_probe exact12l-response-svd-r8-e2e200-offline32 --awq-residual-response-svd

printf '{"status":"COMPLETE","backbone":"awq-w2a16-12l-mixed-spatial-v1","target_blocks":[18,19],"rank":8,"training_steps":200,"objective":"normalized_action_mse_8x7","calibration_trajectories":16,"note":"End-to-end action distillation pilot; offline agreement is not closed-loop success."}\n' > "$OUT_ROOT/e2e-complete.json"
(cd "$OUT_ROOT" && find . -type f ! -name SESSION_SHA256SUMS.txt -print0 | sort -z | xargs -0 -r sha256sum > SESSION_SHA256SUMS.txt)
