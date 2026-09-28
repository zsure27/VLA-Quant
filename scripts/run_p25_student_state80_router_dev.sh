#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=${VLA_EXPERIMENT_ROOT:-/root/autodl-tmp/qvla-repro}
REPO=${VLA_REPO:-/root/autodl-tmp/VLA-Quant-p2c-20260925}
OFT_ROOT=${VLA_OFT_ROOT:-$ROOT/src/QVLA/openvla-oft}
TRAIN=$ROOT/backups/experiments/p2-shared-peft/20260929-107-p25-student-state80-distill/exact12l-response-svd-r8-smoothl1-studentstate80-e2e1000
STATE=$TRAIN/e2e_adapter_state.pt
CALIBRATION=$ROOT/backups/experiments/p1-data-contract/20260926-107-p1-data-contract/peft-train-calibration80
CONTROL=$ROOT/backups/experiments/p2-shared-peft/20260927-107-p2-router-dev
OUT=$ROOT/backups/experiments/p2-shared-peft/20260929-107-p25-student-state80-distill/router-dev-student-state
test -s "$TRAIN/e2e_distillation.json"
test -s "$STATE"
test -s "$CONTROL/router-dev-frames-p10-p30-p50-p70-p90/manifest.json"
test -s "$CONTROL/bf16-teacher/manifest.json"
test -s "$CONTROL/exact12l-baseline/metrics.json"
test ! -e "$OUT"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || exit 4
mkdir -p "$OUT"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/exit-code.txt"' EXIT
export PYTHONPATH="$REPO/diagnostics:$REPO:$OFT_ROOT:$ROOT/src/LIBERO"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=7 WANDB_MODE=disabled
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4
/root/miniconda3/envs/qvla-oft/bin/python -u "$REPO/diagnostics/probe.py" \
  --checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial" \
  --samples-dir "$CONTROL/router-dev-frames-p10-p30-p50-p70-p90" \
  --official-root "$ROOT/src/official-quantization" \
  --targets-file "$REPO/configs/qvla-connected-422.txt" \
  --offset 0 --seed 7 --num-samples 295 --attention-layers '' --action-only \
  --mode awq --weight-bits 2 --activation-bits 16 --weight-scope all \
  --awq-profile "$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w2.pt" \
  --awq-disable-clip attention \
  --awq-w4-layers 8,9,10,11,12,13,14,15,20,21,22,23 \
  --awq-w4-profile "$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w4.pt" \
  --awq-vision-bits 2 --awq-vision-branch all \
  --awq-primary-group64-profile "$ROOT/artifacts/awq-primary-group-20260913-213127-3342/profiles/w2-g64.pt" \
  --awq-exact-attention-visual-stage \
  --awq-residual-layers 18,19 --awq-residual-rank 8 \
  --awq-residual-calibration-dir "$CALIBRATION" --awq-residual-calibration-count 80 \
  --awq-residual-token-scope action --awq-residual-response-svd \
  --awq-recovery-lora-state "$STATE" --teacher-dir "$CONTROL/bf16-teacher" \
  --output "$OUT/candidate" > "$OUT/console.log" 2>&1
sha256sum "$REPO/diagnostics/probe.py" "$REPO/scripts/run_p25_student_state80_router_dev.sh" \
  "$CONTROL/router-dev-frames-p10-p30-p50-p70-p90/manifest.json" \
  "$CONTROL/bf16-teacher/manifest.json" "$STATE" > "$OUT/CONTRACT_SHA256SUMS.txt"
printf '{"status":"COMPLETE","role":"router_dev_trajectory_level","holdout_touched":false}\n' > "$OUT/complete.json"
