#!/usr/bin/env bash
set -Eeuo pipefail
# 每个候选新进程，只保留一份模型显存；不自动启动昂贵 rollout/训练。
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=${ROOT:-/root/autodl-tmp/qvla-repro}
source "$HERE/scripts/activate_oft.sh"
SAMPLES=${SAMPLES:-$ROOT/calib/action-space-balanced/libero-512}
CKPT=${CKPT:-$ROOT/models/openvla-7b-oft-finetuned-libero-spatial}
OFFICIAL=${OFFICIAL:-$ROOT/src/official-quantization}
OUT=${OUT:-$ROOT/artifacts/baseline-audit-$(date +%Y%m%d-%H%M%S)}
CALIB=${CALIB:-8}
N=${N:-8}
OFFSET=${OFFSET:-64}
STAGE=${STAGE:-controls}
export PYTHONPATH="$HERE:$PYTHONPATH"
if [[ "$STAGE" == controls ]]; then
  test ! -e "$OUT" || { echo "拒绝覆盖：$OUT"; exit 1; }
  mkdir -p "$OUT" "$OUT/profiles/sq"
elif [[ "$STAGE" == sq || "$STAGE" == awq ]]; then
  test -f "$OUT/controls.pass.json" || { echo "先完成 controls，设置 OUT 为其输出目录"; exit 1; }
  # 不以文件存在为成功：旧 tee 可能遗留空文件。重新验证真实指标。
  python "$HERE/diagnostics/gates.py" control --metrics "$OUT/repeat/metrics.json" --threshold 1e-10
  python "$HERE/diagnostics/gates.py" control --metrics "$OUT/smooth-only/metrics.json" --threshold 1e-4
else
  echo "STAGE 只能为 controls/sq/awq"; exit 1
fi
python "$HERE/scripts/check_runtime.py" | tee "$OUT/runtime-$STAGE.json"
python "$HERE/diagnostics/gates.py" split --samples "$SAMPLES" --calibration-count "$CALIB" \
  --offset "$OFFSET" --count "$N" | tee "$OUT/split-$STAGE.json"
common=(--checkpoint "$CKPT" --samples-dir "$SAMPLES" --official-root "$OFFICIAL"
        --targets-file "$HERE/configs/qvla-connected-422.txt" --num-samples "$N" --offset "$OFFSET" --seed "${SEED:-7}"
        --attention-layers "${ATTENTION_LAYERS:-}")
calibrate() {
  local method=$1 bits=$2 dest=$3
  python "$HERE/qvla/calibrate_official_quant.py" --method "$method" --bits "$bits" \
    --checkpoint "$CKPT" --samples-dir "$SAMPLES" --official-root "$OFFICIAL" \
    --targets-file "$HERE/configs/qvla-connected-422.txt" --num-samples "$CALIB" --seed "${SEED:-7}" \
    --max-rows "$((CALIB * 32))" --output "$dest.pt" --summary-json "$dest.json" 2>&1 | tee "$dest.log"
}
probe() {
  local name=$1 mode=$2
  shift 2
  python "$HERE/diagnostics/probe.py" "${common[@]}" --mode "$mode" \
    --teacher-dir "$OUT/teacher" --output "$OUT/$name" "$@" 2>&1 | tee "$OUT/$name.log"
}
sq() { local name=$1; shift; probe "$name" smoothquant --profile-dir "$OUT/profiles/sq" "$@"; }
if [[ "$STAGE" == controls ]]; then
  python "$HERE/diagnostics/self_test.py" --official-root "$OFFICIAL" 2>&1 | tee "$OUT/self-test.log"
  calibrate smoothquant 4 "$OUT/profiles/sq/calibration"
  probe teacher teacher
  probe repeat repeat
  python "$HERE/diagnostics/gates.py" control --metrics "$OUT/repeat/metrics.json" --threshold 1e-10 | tee "$OUT/repeat.pass.pending"
  mv "$OUT/repeat.pass.pending" "$OUT/repeat.pass.json"
  sq smooth-only --weight-bits 16 --activation-bits 16
  python "$HERE/diagnostics/gates.py" control --metrics "$OUT/smooth-only/metrics.json" --threshold 1e-4 | tee "$OUT/controls.pass.pending"
  mv "$OUT/controls.pass.pending" "$OUT/controls.pass.json"
elif [[ "$STAGE" == sq ]]; then
  sq sq-w4a16 --activation-bits 16
  sq sq-w16a4 --weight-bits 16
  sq sq-w4a8 --activation-bits 8
  sq sq-w4a4
  sq sq-vision --weight-scope vision --activation-scope vision
  sq sq-language --weight-scope language --activation-scope language
  sq sq-oracle --oracle-projector
  sq sq-vision-oracle --weight-scope vision --activation-scope vision --oracle-projector
elif [[ "$STAGE" == awq ]]; then
  mkdir -p "$OUT/profiles/awq"
  for bits in 4 2; do
    calibrate awq "$bits" "$OUT/profiles/awq/w$bits"
    for scope in all vision language; do
      probe "awq-w$bits-$scope" awq --awq-profile "$OUT/profiles/awq/w$bits.pt" \
        --weight-bits "$bits" --activation-bits 16 --weight-scope "$scope"
    done
  done
  probe awq-w2-oracle awq --awq-profile "$OUT/profiles/awq/w2.pt" \
    --weight-bits 2 --activation-bits 16 --oracle-projector
fi
python "$HERE/diagnostics/plot_results.py" --root "$OUT"
echo "完成 $STAGE：$OUT。请先检查控制结果，再选择下一阶段；不是成功率评估。"
