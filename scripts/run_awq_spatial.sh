#!/usr/bin/env bash
set -Eeuo pipefail
# AWQ 独立路径：不绕过自己的控制，也不依赖 SQ 平滑是否通过。
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$HERE/scripts/activate_oft.sh"
ROOT=${ROOT:-/root/autodl-tmp/qvla-repro}
OUT=${OUT:-$ROOT/artifacts/awq-spatial-$(date +%Y%m%d-%H%M%S)-$$}
STAGE=${STAGE:-w4}
export PYTHONPATH="$HERE:$HERE/diagnostics:$PYTHONPATH"
export PYTHONHASHSEED=7 WANDB_MODE=disabled
CKPT="$ROOT/models/openvla-7b-oft-finetuned-libero-spatial"
OFFICIAL="$ROOT/src/official-quantization"
SAMPLES="$OUT/samples"
if [[ "$STAGE" == w4 ]]; then
  test ! -e "$OUT" || { echo "拒绝覆盖：$OUT"; exit 1; }
  mkdir -p "$OUT/profiles"
  python "$HERE/scripts/prepare_spatial_audit.py" \
    --source "$ROOT/calib/action-space-balanced/libero-512" --output "$SAMPLES"
elif [[ "$STAGE" == w2 ]]; then
  test -s "$OUT/awq-w4-all/metrics.json" || { echo "先完成 W4 并人工审阅，再指定原 OUT"; exit 1; }
else
  echo "STAGE 只能为 w4 或 w2"; exit 1
fi
python "$HERE/scripts/check_runtime.py" | tee "$OUT/runtime-$STAGE.json"
python "$HERE/diagnostics/gates.py" split --samples "$SAMPLES" \
  --calibration-count 32 --offset 32 --count 32 | tee "$OUT/split-$STAGE.json"
common=(--checkpoint "$CKPT" --samples-dir "$SAMPLES" --official-root "$OFFICIAL"
  --targets-file "$HERE/configs/qvla-connected-422.txt" --num-samples 32 --offset 32
  --seed 7 --attention-layers 7,15,23,31 --teacher-dir "$OUT/teacher")
probe() {
  local name=$1; shift
  python "$HERE/diagnostics/probe.py" "${common[@]}" --output "$OUT/$name" "$@" \
    2>&1 | tee "$OUT/$name.log"
}
if [[ "$STAGE" == w4 ]]; then
  python "$HERE/diagnostics/self_test.py" --official-root "$OFFICIAL" | tee "$OUT/self-test.log"
  probe teacher --mode teacher
  probe repeat --mode repeat
fi
# 每次都验证真实 repeat 指标，不依赖空 PASS 文件。
python "$HERE/diagnostics/gates.py" control --metrics "$OUT/repeat/metrics.json" --threshold 1e-10
bits=4
if [[ "$STAGE" == w2 ]]; then bits=2; fi
python "$HERE/qvla/calibrate_official_quant.py" --method awq --bits "$bits" \
  --awq-search official-block --checkpoint "$CKPT" --samples-dir "$SAMPLES" \
  --official-root "$OFFICIAL" --targets-file "$HERE/configs/qvla-connected-422.txt" \
  --num-samples 32 --seed 7 --max-rows 1024 \
  --output "$OUT/profiles/w$bits.pt" --summary-json "$OUT/profiles/w$bits.json" \
  2>&1 | tee "$OUT/profiles/w$bits.log"
scopes=(all)
if [[ "$STAGE" == w2 ]]; then scopes=(all vision language); fi
for scope in "${scopes[@]}"; do
  probe "awq-w$bits-$scope" --mode awq --awq-profile "$OUT/profiles/w$bits.pt" \
    --weight-bits "$bits" --activation-bits 16 --weight-scope "$scope"
done
if [[ "$STAGE" == w2 ]]; then
  probe awq-w2-oracle --mode awq --awq-profile "$OUT/profiles/w2.pt" \
    --weight-bits 2 --activation-bits 16 --oracle-projector
fi
echo "AWQ_SPATIAL_STAGE: COMPLETE $STAGE（不代表成功率或论文复现通过）"
echo "结果目录：$OUT"
