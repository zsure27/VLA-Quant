#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/autodl-tmp/qvla-repro
REPO=/root/VLA-Quant
OVERLAY=$ROOT/overlays/fp32-smoothing-pairs
OLD=$ROOT/artifacts/controls-20260911-234754-1133
OUT=$ROOT/artifacts/sq-fp32-pairs-$(date +%Y%m%d-%H%M%S)-$$
mkdir -p "$OUT"
printf '%s\n' "$OUT" > "$ROOT/artifacts/LATEST_SQ_FP32_PAIRS.txt"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/exit-code.txt"' EXIT
source /root/miniconda3/bin/activate /root/miniconda3/envs/qvla-oft
export PYTHONPATH="$OVERLAY/diagnostics:$REPO:$REPO/diagnostics:$ROOT/src/QVLA/openvla-oft:$ROOT/src/LIBERO"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=7 WANDB_MODE=disabled
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4
cp "$OVERLAY/diagnostics/probe.py" "$OUT/probe-source.py"
cp "$OVERLAY/diagnostics/fp32_smoothing_pairs.py" "$OUT/pair-source.py"
common=(--checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial"
 --samples-dir "$ROOT/calib/action-space-balanced/libero-512"
 --official-root "$ROOT/src/official-quantization" --targets-file "$REPO/configs/qvla-connected-422.txt"
 --num-samples 8 --offset 64 --seed 7 --attention-layers 7,15,23,31
 --weight-bits 16 --activation-bits 16 --mode smoothquant --profile-dir "$OLD/profiles/sq"
 --smoothing-selection no-language --smoothing-pairs-fp32)
run() {
 local name=$1; shift
 printf '%q ' python "$OVERLAY/diagnostics/probe.py" "${common[@]}" "$@" --output "$OUT/$name" > "$OUT/$name.command.txt"
 printf '\n' >> "$OUT/$name.command.txt"
 python "$OVERLAY/diagnostics/probe.py" "${common[@]}" "$@" --output "$OUT/$name" 2>&1 | tee "$OUT/$name.console.log"
}
run fp32-repeat --teacher-dir "$OLD/teacher" --smoothing-bypass --export-candidate-teacher
run fp32-smooth --teacher-dir "$OUT/fp32-repeat" --teacher-fp32-reference
printf '{"status":"MEASUREMENT_COMPLETE","note":"FP32 vision norm/Linear pair control vs BF16, then smoothing vs same FP32 path; W16A16 only, no low-bit/PEFT baseline claim."}\n' > "$OUT/complete.json"
(cd "$OUT" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS.txt)
