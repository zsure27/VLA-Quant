#!/usr/bin/env bash
set -Eeuo pipefail
# 复用G64主视觉profile，只恢复融合视觉W2；不重新校准。
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export ROOT=${ROOT:-/root/autodl-tmp/qvla-repro}
source "$HERE/scripts/activate_oft.sh"
export PYTHONPATH="$HERE:$HERE/diagnostics:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=7 WANDB_MODE=disabled
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4
export BASE=${BASE:-$ROOT/artifacts/awq-spatial-20260912-163735-1136}
export GROUP_RUN=${GROUP_RUN:-$ROOT/artifacts/awq-primary-group-20260913-213127-3342}
export VISION_RUN=${VISION_RUN:-$ROOT/artifacts/awq-vision-branches-20260913-150727-1162}
export OUT=${OUT:-$ROOT/artifacts/awq-group-composition-$(date +%Y%m%d-%H%M%S)-$$}
test -s "$GROUP_RUN/profiles/w2-g64.pt"
test -s "$GROUP_RUN/primary-g64/metrics.json"
test -s "$VISION_RUN/vision-w2-all/metrics.json"
test ! -e "$OUT" || { echo "拒绝覆盖：$OUT"; exit 1; }
mkdir -p "$OUT"
echo "结果目录：$OUT"
python "$HERE/scripts/check_runtime.py" | tee "$OUT/runtime.json"
python "$HERE/diagnostics/gates.py" split --samples "$BASE/samples" --calibration-count 32 --offset 32 --count 32 | tee "$OUT/split.json"
common=(--checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial"
  --samples-dir "$BASE/samples" --official-root "$ROOT/src/official-quantization"
  --targets-file "$HERE/configs/qvla-connected-422.txt" --num-samples 32 --offset 32
  --seed 7 --attention-layers 7,15,23,31 --teacher-dir "$BASE/teacher")
run() {
  local name=$1; shift
  python "$HERE/diagnostics/probe.py" "${common[@]}" --output "$OUT/$name" "$@" 2>&1 | tee "$OUT/$name.log"
}
run repeat --mode repeat
python "$HERE/diagnostics/gates.py" control --metrics "$OUT/repeat/metrics.json" --threshold 1e-10
awq=(--mode awq --weight-bits 2 --activation-bits 16 --awq-profile "$BASE/profiles/w2.pt"
  --awq-disable-clip all --weight-scope all --awq-vision-bits 2)
run all-g128 "${awq[@]}" --awq-vision-branch all
run primary-g64 "${awq[@]}" --awq-vision-branch primary --awq-primary-group64-profile "$GROUP_RUN/profiles/w2-g64.pt"
python "$HERE/scripts/check_group_composition.py" --root "$OUT" --group-run "$GROUP_RUN" --vision-run "$VISION_RUN" --replay-only
run combined "${awq[@]}" --awq-vision-branch all --awq-primary-group64-profile "$GROUP_RUN/profiles/w2-g64.pt"
python "$HERE/scripts/check_group_composition.py" --root "$OUT" --group-run "$GROUP_RUN" --vision-run "$VISION_RUN"
