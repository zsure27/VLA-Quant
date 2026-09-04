#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${ROOT:-/root/autodl-tmp/qvla-repro}
OFT=${OFT:-$ROOT/src/QVLA/openvla-oft}
LIBERO_ROOT=${LIBERO_ROOT:-$ROOT/src/LIBERO}
OFFICIAL=${OFFICIAL:-$ROOT/src/official-quantization}
GROUP_DIR=${GROUP_DIR:-$ROOT/artifacts/awq-vl-w2-target-groups}
CHECKPOINT=${CHECKPOINT:-$ROOT/models/openvla-7b-oft-finetuned-libero-spatial}
SAMPLES=${SAMPLES:-$ROOT/calib/action-space-balanced/libero-512}
OUT_ROOT=${OUT_ROOT:-$ROOT/qvla/spatial/official-w4-vl}
LOG_ROOT=${LOG_ROOT:-$ROOT/logs/official-w4-vl}
SUMMARY_ROOT=${SUMMARY_ROOT:-$ROOT/artifacts/official-w4-vl}
RUNTIME=$SUMMARY_ROOT/runtime-smoke20.csv

CALIBRATOR=$OFT/qvla/calibrate_official_quant.py
RUNNER=$OFT/qvla/run_eval_official_quant.py
TARGET_GROUP_NAMES=(
  00-vision
  language-00-03
  language-04-07
  language-08-11
  language-12-15
  language-16-19
  language-20-23
  language-24-27
  language-28-31
)

source "$ROOT/envs/qvla-oft/bin/activate"
source "$ROOT/env.sh"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

mkdir -p "$OUT_ROOT/awq" "$OUT_ROOT/smoothquant" "$LOG_ROOT" "$SUMMARY_ROOT"
cd "$OFT"

for required in "$CALIBRATOR" "$RUNNER" "$CHECKPOINT" "$SAMPLES"; do
  test -e "$required" || { echo "Missing required path: $required"; exit 1; }
done
for group in "${TARGET_GROUP_NAMES[@]}"; do
  test -s "$GROUP_DIR/$group.txt" || {
    echo "Missing target group: $GROUP_DIR/$group.txt"
    exit 1
  }
done

python - "$GROUP_DIR" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = [root / "00-vision.txt"] + [
    root / f"language-{start:02d}-{start + 3:02d}.txt"
    for start in range(0, 32, 4)
]
names = [line.strip() for path in files for line in path.read_text().splitlines() if line.strip()]
primary = sum(x.startswith("vision_backbone.featurizer.") for x in names)
fused = sum(x.startswith("vision_backbone.fused_featurizer.") for x in names)
language = sum(x.startswith("language_model.") for x in names)
assert len(names) == len(set(names)) == 422
assert (primary, fused, language) == (93, 105, 224)
print(f"targets={len(names)} primary={primary} fused={fused} language={language}")
print("OFFICIAL W4 SCOPE AUDIT: PASS")
PY

if [[ ! -f "$RUNTIME" ]]; then
  echo "method,stage,group,targets,exit_code,wall_seconds" > "$RUNTIME"
fi

profile_valid() {
  python - "$1" "$2" "$3" <<'PY'
import sys
from pathlib import Path
import torch

path, method, expected = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit(1)
try:
    payload = torch.load(path, map_location="cpu")
    valid = (
        payload.get("format_version") == 2
        and payload.get("method") == method
        and payload.get("bits") == 4
        and len(payload.get("entries", {})) == expected
    )
except Exception:
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

for method in awq smoothquant; do
  for group in "${TARGET_GROUP_NAMES[@]}"; do
    target_file=$GROUP_DIR/$group.txt
    target_count=$(grep -cve '^[[:space:]]*$' "$target_file")
    profile=$OUT_ROOT/$method/$group.w4.pt
    summary=$SUMMARY_ROOT/$method-$group-calibration.json
    log=$LOG_ROOT/calibrate-$method-$group.log

    if profile_valid "$profile" "$method" "$target_count"; then
      echo "[resume] valid profile: $profile"
      continue
    fi

    start=$(date +%s)
    set +e
    CUDA_VISIBLE_DEVICES=0 python "$CALIBRATOR" \
      --method "$method" \
      --checkpoint "$CHECKPOINT" \
      --samples-dir "$SAMPLES" \
      --targets-file "$target_file" \
      --official-root "$OFFICIAL" \
      --output "$profile" \
      --summary-json "$summary" \
      --num-samples 32 \
      --bits 4 \
      2>&1 | tee "$log"
    status=${PIPESTATUS[0]}
    set -e
    elapsed=$(($(date +%s) - start))
    echo "$method,calibration,$group,$target_count,$status,$elapsed" >> "$RUNTIME"
    [[ $status -eq 0 ]] || exit "$status"
    profile_valid "$profile" "$method" "$target_count" || {
      echo "Invalid generated profile: $profile"
      exit 1
    }
  done
done

run_evaluation() {
  local method=$1
  local log=$LOG_ROOT/$method-w4-smoke20.log
  local eval_dir=$ROOT/eval/official-$method-w4-smoke20
  local profile_args=()
  local group
  for group in "${TARGET_GROUP_NAMES[@]}"; do
    profile_args+=(--profile "$OUT_ROOT/$method/$group.w4.pt")
  done

  local start status elapsed
  start=$(date +%s)
  set +e
  CUDA_VISIBLE_DEVICES=0 python "$RUNNER" \
    --method "$method" \
    --weight-bits 4 \
    --activation-bits "$([[ $method == awq ]] && echo 16 || echo 4)" \
    --pretrained_checkpoint "$CHECKPOINT" \
    "${profile_args[@]}" \
    --official-root "$OFFICIAL" \
    --task_suite_name libero_spatial \
    --num_trials_per_task 2 \
    --local_log_dir "$eval_dir" \
    --libero_root "$LIBERO_ROOT" \
    --seed 7 \
    2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  set -e
  elapsed=$(($(date +%s) - start))
  echo "$method,evaluation,all,422,$status,$elapsed" >> "$RUNTIME"
  [[ $status -eq 0 ]] || return "$status"
}

run_evaluation awq
run_evaluation smoothquant

{
  echo "configuration=official-source OpenVLA W4 smoke20"
  echo "quantized_scope=primary vision 93 + fused vision 105 + language 224"
  echo "bf16_scope=projector,proprio_projector,action_head,embeddings,norms"
  for method in awq smoothquant; do
    log=$LOG_ROOT/$method-w4-smoke20.log
    echo "===== $method ====="
    grep -E "Total episodes|Total successes|Overall success rate" "$log"
  done
} | tee "$SUMMARY_ROOT/smoke20-summary.txt"

echo "OFFICIAL W4 SMOKE20: PASS"
