#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/autodl-tmp/qvla-repro
REPO=/root/VLA-Quant
OVERLAY=$ROOT/overlays/language-family-rescue
OUT=$ROOT/artifacts/sq-local-equivalence-$(date +%Y%m%d-%H%M%S)-$$
mkdir -p "$OUT"
printf '%s\n' "$OUT" > "$ROOT/artifacts/LATEST_SQ_LOCAL_EQUIVALENCE.txt"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/exit-code.txt"' EXIT
source /root/miniconda3/bin/activate /root/miniconda3/envs/qvla-oft
export PYTHONPATH="$OVERLAY/diagnostics:$REPO:$REPO/diagnostics:$ROOT/src/QVLA/openvla-oft:$ROOT/src/LIBERO"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=7 WANDB_MODE=disabled
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4
cp "$OVERLAY/diagnostics/sq_local_equivalence.py" "$OUT/source.py"
python "$OVERLAY/diagnostics/sq_local_equivalence.py" \
 --checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial" \
 --profile "$ROOT/artifacts/controls-20260911-234754-1133/profiles/sq/calibration.pt" \
 --sample "$ROOT/calib/action-space-balanced/libero-512/sample-0065.npz" \
 --official-root "$ROOT/src/official-quantization" --output "$OUT/measurement" 2>&1 | tee "$OUT/console.log"
(cd "$OUT" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS.txt)
