#!/usr/bin/env bash
set -Eeuo pipefail
# 原样上传此目录；脚本不会覆盖现有源码、profile 或检查点。
ROOT=${ROOT:-/root/autodl-tmp/qvla-repro}
OFT=${OFT:-$ROOT/src/QVLA/openvla-oft}
DIAG=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:-$ROOT/artifacts/vla-quant-audit-$(date +%Y%m%d-%H%M%S)}
SAMPLES=${SAMPLES:-$ROOT/calib/action-space-balanced/libero-512}
OFFICIAL=${OFFICIAL:-$ROOT/src/official-quantization}
PROFILE=${PROFILE:-$ROOT/qvla/spatial/official-w4-vl/smoothquant}
CKPT=${CKPT:-$ROOT/models/openvla-7b-oft-finetuned-libero-spatial}
TARGETS=${TARGETS:-$DIAG/../configs/qvla-connected-422.txt}
N=${N:-8}
OFFSET=${OFFSET:-64}
source "$ROOT/envs/qvla-oft/bin/activate"
source "$ROOT/env.sh"
export PYTHONPATH="$OFT:$ROOT/src/LIBERO:${PYTHONPATH:-}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
test ! -e "$OUT" || { echo "Output already exists: $OUT"; exit 1; }
mkdir -p "$OUT"
cd "$OFT"
python "$DIAG/self_test.py" --official-root "$OFFICIAL" 2>&1 | tee "$OUT/self-test.log"
common=(--checkpoint "$CKPT" --samples-dir "$SAMPLES" --official-root "$OFFICIAL"
        --targets-file "$TARGETS" --num-samples "$N" --offset "$OFFSET")
python "$DIAG/probe.py" "${common[@]}" --mode awq-micro --output "$OUT/awq-micro" 2>&1 | tee "$OUT/awq-micro.log"
python "$DIAG/probe.py" "${common[@]}" --mode teacher --output "$OUT/teacher" 2>&1 | tee "$OUT/teacher.log"
run_sq() {
  local name=$1
  shift
  python "$DIAG/probe.py" "${common[@]}" --mode smoothquant --profile-dir "$PROFILE" \
    --teacher-dir "$OUT/teacher" --output "$OUT/$name" "$@" 2>&1 | tee "$OUT/$name.log"
}
# 完成这些控制实验后先检查结果，再决定是否运行 500 episode 评测。
run_sq smooth-only --weight-bits 16 --activation-bits 16
run_sq sq-w4a16 --weight-bits 4 --activation-bits 16
run_sq sq-w16a4 --weight-bits 16 --activation-bits 4
run_sq sq-w4a8 --weight-bits 4 --activation-bits 8
run_sq sq-w4a4 --weight-bits 4 --activation-bits 4
run_sq sq-w4a4-vision --weight-scope vision --activation-scope vision
run_sq sq-w4a4-language --weight-scope language --activation-scope language
run_sq sq-w4a4-oracle-projector --oracle-projector
run_sq sq-w4a4-vision-oracle-projector --weight-scope vision --activation-scope vision --oracle-projector
python "$DIAG/plot_results.py" --root "$OUT" 2>&1 | tee "$OUT/plot.log"
echo "Offline diagnostics complete: $OUT"
echo "Send manifest.json, scope.json, awq_micro.json, metrics.json, figures and logs. Teacher .pt caches are not needed."
