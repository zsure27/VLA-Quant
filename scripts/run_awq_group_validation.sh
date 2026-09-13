#!/usr/bin/env bash
set -Eeuo pipefail
# 固定候选，复用之前的独立验证轨迹及教师，不校准、不训练。
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export ROOT=${ROOT:-/root/autodl-tmp/qvla-repro}
source "$HERE/scripts/activate_oft.sh"
export PYTHONPATH="$HERE:$HERE/diagnostics:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=7 WANDB_MODE=disabled
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4
export BASE=${BASE:-$ROOT/artifacts/awq-spatial-20260912-163735-1136}
export GROUP_RUN=${GROUP_RUN:-$ROOT/artifacts/awq-primary-group-20260913-213127-3342}
export VALIDATION=${VALIDATION:-$ROOT/artifacts/awq-validation-20260913-132827-1162}
export OUT=${OUT:-$ROOT/artifacts/awq-group-validation-$(date +%Y%m%d-%H%M%S)-$$}
test ! -e "$OUT" || { echo "拒绝覆盖：$OUT"; exit 1; }
mkdir -p "$OUT"
echo "结果目录：$OUT"
python "$HERE/scripts/audit_group_validation.py" --preflight | tee "$OUT/frozen-inputs.json"
python "$HERE/scripts/check_runtime.py" | tee "$OUT/runtime.json"
common=(--checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial"
  --samples-dir "$VALIDATION/samples" --official-root "$ROOT/src/official-quantization"
  --targets-file "$HERE/configs/qvla-connected-422.txt" --num-samples 32 --offset 0
  --seed 7 --attention-layers 7,15,23,31 --teacher-dir "$VALIDATION/teacher")
run() {
  local name=$1; shift
  python "$HERE/diagnostics/probe.py" "${common[@]}" --output "$OUT/$name" "$@" 2>&1 | tee "$OUT/$name.log"
}
run repeat --mode repeat
python "$HERE/diagnostics/gates.py" control --metrics "$OUT/repeat/metrics.json" --threshold 1e-10
awq=(--mode awq --weight-bits 2 --activation-bits 16 --awq-profile "$BASE/profiles/w2.pt"
  --awq-disable-clip all --weight-scope all --awq-vision-bits 2 --awq-vision-branch all)
run all-g128 "${awq[@]}"
run combined "${awq[@]}" --awq-primary-group64-profile "$GROUP_RUN/profiles/w2-g64.pt"
python "$HERE/scripts/audit_group_validation.py"
