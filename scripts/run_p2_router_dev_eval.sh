#!/usr/bin/env bash
set -Eeuo pipefail

MODE=${1:?usage: run_p2_router_dev_eval.sh prepare | candidate NAME STATE CALIBRATION COUNT}
ROOT=${VLA_EXPERIMENT_ROOT:-/root/autodl-tmp/qvla-repro}
REPO=${VLA_REPO:-/root/autodl-tmp/VLA-Quant-p2c-20260925}
SESSION=${VLA_SESSION:-20260926-107-p2-router-dev}
OUT=$ROOT/backups/experiments/p2-shared-peft/$SESSION
SPLIT=${VLA_TRAJECTORY_SPLIT:-$ROOT/backups/experiments/p1-data-contract/20260925-107-p1-data-contract/trajectory-split.json}
DATASET=${VLA_SPATIAL_DATASET:-$ROOT/datasets/libero_spatial_no_noops/1.0.0}
SAMPLES=$OUT/router-dev-frames-p10-p30-p50-p70-p90
TEACHER=$OUT/bf16-teacher
BASELINE=$OUT/exact12l-baseline
BASE=$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w2.pt
W4=$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w4.pt
G64=$ROOT/artifacts/awq-primary-group-20260913-213127-3342/profiles/w2-g64.pt
OFT_ROOT=${VLA_OFT_ROOT:-$ROOT/src/QVLA/openvla-oft}

source /root/miniconda3/bin/activate /root/miniconda3/envs/qvla-oft
export PYTHONPATH="$REPO/diagnostics:$REPO:$OFT_ROOT:$ROOT/src/LIBERO"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=7 WANDB_MODE=disabled
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4

common=(
  --checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial"
  --samples-dir "$SAMPLES" --official-root "$ROOT/src/official-quantization"
  --targets-file "$REPO/configs/qvla-connected-422.txt" --offset 0 --seed 7
  --num-samples 295 --attention-layers '' --action-only
)
exact12=(
  --mode awq --weight-bits 2 --activation-bits 16 --weight-scope all
  --awq-profile "$BASE" --awq-disable-clip attention
  --awq-w4-layers 8,9,10,11,12,13,14,15,20,21,22,23 --awq-w4-profile "$W4"
  --awq-vision-bits 2 --awq-vision-branch all --awq-primary-group64-profile "$G64"
  --awq-exact-attention-visual-stage
)

if [[ "$MODE" == prepare ]]; then
  test ! -e "$OUT"
  mkdir -p "$OUT"
  python "$REPO/qvla/extract_trajectory_frames.py" --dataset-directory "$DATASET" \
    --trajectory-split "$SPLIT" --role router_dev --positions 0.1,0.3,0.5,0.7,0.9 --output "$SAMPLES"
  python -u "$REPO/diagnostics/probe.py" "${common[@]}" --mode teacher --output "$TEACHER" \
    2>&1 | tee "$OUT/bf16-teacher-console.log"
  python -u "$REPO/diagnostics/probe.py" "${common[@]}" "${exact12[@]}" --teacher-dir "$TEACHER" --output "$BASELINE" \
    2>&1 | tee "$OUT/exact12l-baseline-console.log"
elif [[ "$MODE" == candidate ]]; then
  NAME=${2:?candidate name required}; STATE=${3:?state required}; CALIBRATION=${4:?calibration required}; COUNT=${5:?count required}
  [[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || exit 2
  test -s "$STATE"; test -d "$CALIBRATION"; test -s "$TEACHER/manifest.json"; test -s "$BASELINE/metrics.json"
  CANDIDATE=$OUT/$NAME
  test ! -e "$CANDIDATE"
  python -u "$REPO/diagnostics/probe.py" "${common[@]}" "${exact12[@]}" --teacher-dir "$TEACHER" \
    --awq-residual-layers 18,19 --awq-residual-rank 8 --awq-residual-calibration-dir "$CALIBRATION" \
    --awq-residual-calibration-count "$COUNT" --awq-residual-token-scope action --awq-residual-response-svd \
    --awq-recovery-lora-state "$STATE" --output "$CANDIDATE" 2>&1 | tee "$OUT/$NAME-console.log"
else
  echo "Unknown mode: $MODE" >&2; exit 2
fi
(cd "$OUT" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS.txt)

