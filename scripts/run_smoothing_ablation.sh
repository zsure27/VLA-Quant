#!/usr/bin/env bash
set -Eeuo pipefail
# 定位专用入口，不生成 controls.pass.json，也不放行低比特批量实验。
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$HERE/scripts/activate_oft.sh"
ROOT=${ROOT:-/root/autodl-tmp/qvla-repro}
BASE=${BASE:-$ROOT/artifacts/controls-20260911-234754-1133}
OUT=${OUT:-$ROOT/artifacts/smoothing-ablation-$(date +%Y%m%d-%H%M%S)-$$}
export PYTHONPATH="$HERE:$HERE/diagnostics:$PYTHONPATH"
export WANDB_MODE=disabled
export PYTHONHASHSEED=7
test ! -e "$OUT" || { echo "拒绝覆盖：$OUT"; exit 1; }
test -f "$BASE/teacher/manifest.json"
test -f "$BASE/profiles/sq/calibration.pt"
mkdir -p "$OUT"
common=(--checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial"
  --samples-dir "$ROOT/calib/action-space-balanced/libero-512"
  --official-root "$ROOT/src/official-quantization"
  --targets-file "$HERE/configs/qvla-connected-422.txt"
  --teacher-dir "$BASE/teacher" --num-samples 8 --offset 64 --seed 7
  --attention-layers 7,15,23,31 --weight-bits 16 --activation-bits 16)
python "$HERE/scripts/check_runtime.py" | tee "$OUT/runtime.json"
python "$HERE/diagnostics/probe.py" "${common[@]}" --mode repeat \
  --output "$OUT/repeat" 2>&1 | tee "$OUT/repeat.log"
python "$HERE/diagnostics/gates.py" control --metrics "$OUT/repeat/metrics.json" --threshold 1e-10
for selection in all no-late-attention no-late-mlp no-late-both; do
  python "$HERE/diagnostics/probe.py" "${common[@]}" --mode smoothquant \
    --profile-dir "$BASE/profiles/sq" --oracle-projector \
    --smoothing-selection "$selection" --output "$OUT/$selection" \
    2>&1 | tee "$OUT/$selection.log"
done
echo "SMOOTHING_ABLATION: COMPLETE（定位实验完成，不代表控制门槛通过）"
echo "输出目录：$OUT"
