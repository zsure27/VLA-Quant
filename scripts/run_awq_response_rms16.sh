#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/autodl-tmp/qvla-repro
REPO=/root/VLA-Quant
OVERLAY=$ROOT/overlays/response-svd-recovery
BASE=$ROOT/artifacts/awq-spatial-20260912-163735-1136
VALIDATION=$ROOT/artifacts/awq-validation-20260913-132827-1162
OUT=$ROOT/artifacts/awq-response-svd-rms16-$(date +%Y%m%d-%H%M%S)-$$
mkdir -p "$OUT"
printf '%s\n' "$OUT" > "$ROOT/artifacts/LATEST_RESPONSE_SVD_RMS16.txt"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/exit-code.txt"' EXIT
source /root/miniconda3/bin/activate /root/miniconda3/envs/qvla-oft
export PYTHONPATH="$OVERLAY/diagnostics:$REPO:$REPO/diagnostics:$ROOT/src/QVLA/openvla-oft:$ROOT/src/LIBERO"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=7 WANDB_MODE=disabled
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4
git -C "$REPO" rev-parse HEAD > "$OUT/base-commit.txt"
cp "$OVERLAY/diagnostics/probe.py" "$OUT/probe-source.py"
cp "$OVERLAY/diagnostics/low_rank_recovery.py" "$OUT/low-rank-source.py"
common=(--checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial"
 --samples-dir "$VALIDATION/samples" --official-root "$ROOT/src/official-quantization"
 --targets-file "$REPO/configs/qvla-connected-422.txt" --num-samples 32 --offset 0
 --seed 7 --attention-layers 7,15,23,31 --teacher-dir "$VALIDATION/teacher"
 --mode awq --weight-bits 2 --activation-bits 16 --weight-scope language
 --awq-profile "$BASE/profiles/w2.pt" --awq-disable-clip all
 --awq-residual-layers 8,9,10,11,12,13,14,15 --awq-residual-calibration-dir "$ROOT/calib/action-space-balanced/libero-512")
for rank in 16; do
 printf '%q ' python "$OVERLAY/diagnostics/probe.py" "${common[@]}" --awq-residual-token-scope action --awq-residual-rank "$rank" --output "$OUT/rank-$rank" > "$OUT/rank-$rank.command.txt"
 printf '\n' >> "$OUT/rank-$rank.command.txt"
 python "$OVERLAY/diagnostics/probe.py" "${common[@]}" --awq-residual-token-scope action --awq-residual-rank "$rank" --output "$OUT/rank-$rank" 2>&1 | tee "$OUT/rank-$rank.console.log"
done
printf '{"status":"MEASUREMENT_COMPLETE","training_steps":0,"note":"One matched rank16 action-RMS SVD control; frozen W2, no rollout, no trained PEFT claim."}\n' > "$OUT/complete.json"
(cd "$OUT" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS.txt)
