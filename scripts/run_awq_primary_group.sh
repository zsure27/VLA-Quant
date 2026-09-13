#!/usr/bin/env bash
set -Eeuo pipefail
# 完整校准G64，仅在推理中取主视觉条目；不替换语言profile。
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export ROOT=${ROOT:-/root/autodl-tmp/qvla-repro}
source "$HERE/scripts/activate_oft.sh"
export PYTHONPATH="$HERE:$HERE/diagnostics:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=7 WANDB_MODE=disabled
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4
export BASE=${BASE:-$ROOT/artifacts/awq-spatial-20260912-163735-1136}
export PREVIOUS=${PREVIOUS:-$ROOT/artifacts/awq-primary-clip-20260913-203607-1400}
export OUT=${OUT:-$ROOT/artifacts/awq-primary-group-$(date +%Y%m%d-%H%M%S)-$$}
test -s "$PREVIOUS/primary-original/metrics.json"
test ! -e "$OUT" || { echo "拒绝覆盖：$OUT"; exit 1; }
mkdir -p "$OUT/profiles"
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
  --awq-disable-clip all --weight-scope all --awq-vision-bits 2 --awq-vision-branch primary)
run primary-g128 "${awq[@]}"
python - <<'PY'
import json, os, math
from pathlib import Path
a=Path(os.environ['PREVIOUS'])/'primary-original'
b=Path(os.environ['OUT'])/'primary-g128'
old,new=[{r['sample']:r for r in json.loads((p/'metrics.json').read_text())} for p in (a,b)]
assert set(old)==set(new) and len(new)==32, '参照样本变化'
for k in new:
    x,y=[r[k]['normalized_action']['mse'] for r in (old,new)]
    assert math.isfinite(x) and math.isfinite(y) and abs(x-y)<=1e-8, '参照动作未复现'
    assert old[k]['raw_gripper_disagreement']==new[k]['raw_gripper_disagreement'], '参照夹爪未复现'
assert json.loads((a/'manifest.json').read_text())['profiles']==json.loads((b/'manifest.json').read_text())['profiles'], '参照profile变化'
print('GROUP_REFERENCE_REPLAY: PASS')
PY
python "$HERE/qvla/calibrate_official_quant.py" --method awq --bits 2 --group-size 64 \
  --awq-search official-block --checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial" \
  --samples-dir "$BASE/samples" --official-root "$ROOT/src/official-quantization" \
  --targets-file "$HERE/configs/qvla-connected-422.txt" --num-samples 32 --seed 7 \
  --max-rows 1024 --max-output-rows 256 --clip-tokens 128 \
  --output "$OUT/profiles/w2-g64.pt" --summary-json "$OUT/profiles/w2-g64.json" \
  2>&1 | tee "$OUT/profiles/w2-g64.log"
run primary-g64 "${awq[@]}" --awq-primary-group64-profile "$OUT/profiles/w2-g64.pt"
python "$HERE/scripts/summarize_primary_group.py" "$OUT"
