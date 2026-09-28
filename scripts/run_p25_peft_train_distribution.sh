#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=${VLA_EXPERIMENT_ROOT:-/root/autodl-tmp/qvla-repro}
REPO=${VLA_REPO:-/root/autodl-tmp/VLA-Quant-p2c-20260925}
OFT_ROOT=${VLA_OFT_ROOT:-$ROOT/src/QVLA/openvla-oft}
OUT_ROOT=$ROOT/backups/experiments/p2-shared-peft/20260928-107-p25-distribution-triad
OUT=$OUT_ROOT/peft-train-other-frame80
SAMPLES=$OUT_ROOT/peft-train-frames-p20
CALIBRATION=$ROOT/backups/experiments/p1-data-contract/20260926-107-p1-data-contract/peft-train-calibration80
SPLIT=$ROOT/backups/experiments/p1-data-contract/20260925-107-p1-data-contract/trajectory-inventory-v3/trajectory_split.json
DATASET=$ROOT/data/modified_libero_rlds/libero_spatial_no_noops/1.0.0
STATE=$ROOT/backups/experiments/p2-shared-peft/20260926-107-p2-robust-e2e-distill-data80/exact12l-response-svd-r8-smoothl1-data80-e2e1000-offline32/recovery_adapter_state.pt
BASE=$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w2.pt
W4=$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w4.pt
G64=$ROOT/artifacts/awq-primary-group-20260913-213127-3342/profiles/w2-g64.pt
test ! -e "$OUT"
test -s "$STATE"; test -d "$CALIBRATION"; test -s "$SPLIT"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || exit 4
mkdir -p "$OUT_ROOT"
export PYTHONPATH="$REPO/diagnostics:$REPO:$OFT_ROOT:$ROOT/src/LIBERO"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=7 WANDB_MODE=disabled
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4
/root/miniconda3/envs/qvla-oft/bin/python "$REPO/qvla/extract_trajectory_frames.py" \
  --dataset-directory "$DATASET" --trajectory-split "$SPLIT" --role peft_train \
  --positions 0.2 --output "$SAMPLES"
test -s "$SAMPLES/manifest.json"
mkdir -p "$OUT"
common=(--checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial"
  --samples-dir "$SAMPLES" --official-root "$ROOT/src/official-quantization"
  --targets-file "$REPO/configs/qvla-connected-422.txt" --offset 0 --seed 7
  --num-samples 80 --attention-layers '' --action-only)
exact12=(--mode awq --weight-bits 2 --activation-bits 16 --weight-scope all
  --awq-profile "$BASE" --awq-disable-clip attention
  --awq-w4-layers 8,9,10,11,12,13,14,15,20,21,22,23 --awq-w4-profile "$W4"
  --awq-vision-bits 2 --awq-vision-branch all --awq-primary-group64-profile "$G64"
  --awq-exact-attention-visual-stage)
/root/miniconda3/envs/qvla-oft/bin/python -u "$REPO/diagnostics/probe.py" "${common[@]}" \
  --mode teacher --output "$OUT/bf16-teacher" > "$OUT/bf16-console.log" 2>&1
/root/miniconda3/envs/qvla-oft/bin/python -u "$REPO/diagnostics/probe.py" "${common[@]}" \
  "${exact12[@]}" --teacher-dir "$OUT/bf16-teacher" --output "$OUT/exact12l" > "$OUT/exact12l-console.log" 2>&1
/root/miniconda3/envs/qvla-oft/bin/python -u "$REPO/diagnostics/probe.py" "${common[@]}" \
  "${exact12[@]}" --teacher-dir "$OUT/bf16-teacher" \
  --awq-residual-layers 18,19 --awq-residual-rank 8 --awq-residual-calibration-dir "$CALIBRATION" \
  --awq-residual-calibration-count 80 --awq-residual-token-scope action --awq-residual-response-svd \
  --awq-recovery-lora-state "$STATE" --output "$OUT/12l-lora" > "$OUT/lora-console.log" 2>&1
sha256sum "$REPO/diagnostics/probe.py" "$STATE" "$SAMPLES/manifest.json" \
  "$CALIBRATION/manifest.json" "$SPLIT" > "$OUT/CONTRACT_SHA256SUMS.txt"
printf '{"status":"COMPLETE","role":"peft_train","n":80,"holdout_touched":false}\n' > "$OUT/complete.json"
