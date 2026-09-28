#!/usr/bin/env bash
set -Eeuo pipefail
OFFSET=${1:?offset required}; COUNT=${2:?count required}
[[ "$OFFSET" =~ ^[0-9]+$ && "$COUNT" =~ ^[0-9]+$ ]] || exit 2
(( COUNT == 10 && OFFSET % 10 == 0 && OFFSET + COUNT <= 50 )) || exit 2
ROOT=${VLA_EXPERIMENT_ROOT:-/root/autodl-tmp/qvla-repro}
REPO=${VLA_REPO:-/root/autodl-tmp/VLA-Quant-p2c-20260925}
OFT_ROOT=${VLA_OFT_ROOT:-$ROOT/src/QVLA/openvla-oft}
END=$((OFFSET + COUNT - 1))
SOURCE=$ROOT/backups/experiments/p2-shared-peft/20260928-107-p25-query3-no-lora/control-${OFFSET}-${END}
TRAIN=$ROOT/backups/experiments/p2-shared-peft/20260929-107-p25-student-state80-distill/exact12l-response-svd-r8-smoothl1-studentstate80-e2e1000
CALIBRATION=$ROOT/backups/experiments/p1-data-contract/20260926-107-p1-data-contract/peft-train-calibration80
OUT=$ROOT/backups/experiments/p2-shared-peft/20260929-107-p25-student-state80-distill/query3-student-state-${OFFSET}-${END}
test -s "$SOURCE/samples/manifest.json"
test -s "$SOURCE/bf16-teacher/manifest.json"
test -s "$TRAIN/e2e_adapter_state.pt"
test ! -e "$OUT"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || exit 4
mkdir -p "$OUT"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/exit-code.txt"' EXIT
export PYTHONPATH="$REPO/diagnostics:$REPO:$OFT_ROOT:$ROOT/src/LIBERO"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=7 WANDB_MODE=disabled
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4
/root/miniconda3/envs/qvla-oft/bin/python -u "$REPO/diagnostics/probe.py" \
  --checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial" \
  --samples-dir "$SOURCE/samples" --official-root "$ROOT/src/official-quantization" \
  --targets-file "$REPO/configs/qvla-connected-422.txt" \
  --offset 0 --seed 7 --num-samples 100 --attention-layers '' --action-only \
  --mode awq --weight-bits 2 --activation-bits 16 --weight-scope all \
  --awq-profile "$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w2.pt" \
  --awq-disable-clip attention --awq-w4-layers 8,9,10,11,12,13,14,15,20,21,22,23 \
  --awq-w4-profile "$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w4.pt" \
  --awq-vision-bits 2 --awq-vision-branch all \
  --awq-primary-group64-profile "$ROOT/artifacts/awq-primary-group-20260913-213127-3342/profiles/w2-g64.pt" \
  --awq-exact-attention-visual-stage --teacher-dir "$SOURCE/bf16-teacher" \
  --awq-residual-layers 18,19 --awq-residual-rank 8 \
  --awq-residual-calibration-dir "$CALIBRATION" --awq-residual-calibration-count 80 \
  --awq-residual-token-scope action --awq-residual-response-svd \
  --awq-recovery-lora-state "$TRAIN/e2e_adapter_state.pt" \
  --output "$OUT/candidate" > "$OUT/console.log" 2>&1
sha256sum "$REPO/diagnostics/probe.py" "$REPO/scripts/run_p25_student_state80_query3_control.sh" \
  "$SOURCE/samples/manifest.json" "$SOURCE/bf16-teacher/manifest.json" \
  "$TRAIN/e2e_adapter_state.pt" > "$OUT/CONTRACT_SHA256SUMS.txt"
printf '{"status":"COMPLETE","role":"student_visited_query3","evaluation_reset_filter":"5-49_only","holdout_touched":false}\n' > "$OUT/complete.json"
