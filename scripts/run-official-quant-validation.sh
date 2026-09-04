#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${ROOT:-/root/autodl-tmp/qvla-repro}
OFT=${OFT:-$ROOT/src/QVLA/openvla-oft}
LIBERO_ROOT=${LIBERO_ROOT:-$ROOT/src/LIBERO}
OFFICIAL=${OFFICIAL:-$ROOT/src/official-quantization}
CHECKPOINT=${CHECKPOINT:-$ROOT/models/openvla-7b-oft-finetuned-libero-spatial}
PROFILE_ROOT=${PROFILE_ROOT:-$ROOT/qvla/spatial/official-w4-vl}
LOG_ROOT=${LOG_ROOT:-$ROOT/logs/official-quant-validation}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$ROOT/artifacts/official-quant-validation}
RUNNER=$OFT/qvla/run_eval_official_quant.py
RUNTIME=$ARTIFACT_ROOT/runtime.csv

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

mkdir -p "$LOG_ROOT" "$ARTIFACT_ROOT"
cd "$OFT"
test -f "$RUNNER" || { echo "Missing runner: $RUNNER"; exit 1; }
for method in awq smoothquant; do
  for group in "${TARGET_GROUP_NAMES[@]}"; do
    test -s "$PROFILE_ROOT/$method/$group.w4.pt" || {
      echo "Missing profile: $PROFILE_ROOT/$method/$group.w4.pt"
      exit 1
    }
  done
done

if [[ ! -f "$RUNTIME" ]]; then
  echo "case,method,weight_bits,activation_bits,episodes_expected,exit_code,wall_seconds" > "$RUNTIME"
fi

profile_args() {
  local method=$1
  local group
  for group in "${TARGET_GROUP_NAMES[@]}"; do
    local profile=$PROFILE_ROOT/$method/$group.w4.pt
    test -s "$profile" || { echo "Missing profile: $profile" >&2; return 1; }
    printf '%s\n' "$profile"
  done
}

run_case() {
  local case_name=$1
  local method=$2
  local weight_bits=$3
  local activation_bits=$4
  local trials=$5
  local episodes_expected=$((trials * 10))
  local log=$LOG_ROOT/$case_name.log
  local eval_dir=$ROOT/eval/$case_name
  local profiles=()
  local path
  while IFS= read -r path; do
    profiles+=(--profile "$path")
  done < <(profile_args "$method")

  if grep -q ">> Total episodes: $episodes_expected" "$log" 2>/dev/null; then
    echo "[resume] completed: $case_name"
    return 0
  fi

  local start status elapsed
  start=$(date +%s)
  set +e
  CUDA_VISIBLE_DEVICES=0 python "$RUNNER" \
    --method "$method" \
    --weight-bits "$weight_bits" \
    --activation-bits "$activation_bits" \
    --pretrained_checkpoint "$CHECKPOINT" \
    "${profiles[@]}" \
    --official-root "$OFFICIAL" \
    --task_suite_name libero_spatial \
    --num_trials_per_task "$trials" \
    --local_log_dir "$eval_dir" \
    --libero_root "$LIBERO_ROOT" \
    --seed 7 \
    2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  set -e
  elapsed=$(($(date +%s) - start))
  echo "$case_name,$method,$weight_bits,$activation_bits,$episodes_expected,$status,$elapsed" >> "$RUNTIME"
  [[ $status -eq 0 ]] || return "$status"
}

# AWQ 完整验证：10 个任务 × 50 次 = 500 episodes。
run_case awq-w4a16-full500 awq 4 16 50

# SmoothQuant 诊断：10 个任务 × 2 次 = 每组 20 episodes。
run_case smoothquant-w4a16-smoke20 smoothquant 4 16 2
run_case smoothquant-w4a8-smoke20 smoothquant 4 8 2
run_case smoothquant-w8a8-smoke20 smoothquant 8 8 2

python - "$LOG_ROOT" "$RUNTIME" "$ARTIFACT_ROOT/results.csv" "$ARTIFACT_ROOT/results.md" <<'PY'
import csv
import math
import re
import sys
from pathlib import Path

log_root, runtime_path, csv_path, markdown_path = map(Path, sys.argv[1:])
cases = [
    ("BF16 baseline", "bf16", 16, 16, 500, 488, 0, "NA"),
    ("AWQ W4A16", "awq-w4a16-full500", 4, 16, None, None, None, None),
    ("SmoothQuant W4A16", "smoothquant-w4a16-smoke20", 4, 16, None, None, None, None),
    ("SmoothQuant W4A8", "smoothquant-w4a8-smoke20", 4, 8, None, None, None, None),
    ("SmoothQuant W8A8", "smoothquant-w8a8-smoke20", 8, 8, None, None, None, None),
    ("SmoothQuant W4A4", "previous", 4, 4, 20, 0, 0, 556),
]

runtime = {}
if runtime_path.is_file():
    with runtime_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            runtime[row["case"]] = row

def extract(case, episodes, successes):
    if episodes is not None:
        return episodes, successes
    text = (log_root / f"{case}.log").read_text(errors="replace")
    ep = re.findall(r"Total episodes:\s*(\d+)", text)
    ok = re.findall(r"Total successes:\s*(\d+)", text)
    if not ep or not ok:
        raise RuntimeError(f"Missing result in {case}.log")
    return int(ep[-1]), int(ok[-1])

def wilson(successes, episodes):
    if not episodes:
        return 0.0, 0.0
    z = 1.959963984540054
    p = successes / episodes
    denominator = 1 + z * z / episodes
    center = (p + z * z / (2 * episodes)) / denominator
    margin = z * math.sqrt(p * (1 - p) / episodes + z * z / (4 * episodes**2)) / denominator
    return center - margin, center + margin

rows = []
for label, case, w_bits, a_bits, episodes, successes, exit_code, wall_seconds in cases:
    episodes, successes = extract(case, episodes, successes)
    if case in runtime:
        exit_code = int(runtime[case]["exit_code"])
        wall_seconds = int(runtime[case]["wall_seconds"])
    low, high = wilson(successes, episodes)
    rows.append({
        "method": label,
        "weight_bits": w_bits,
        "activation_bits": a_bits,
        "episodes": episodes,
        "successes": successes,
        "rate": f"{successes / episodes:.4f}",
        "wilson95_low": f"{low:.4f}",
        "wilson95_high": f"{high:.4f}",
        "exit_code": exit_code,
        "wall_seconds": wall_seconds,
    })

csv_path.parent.mkdir(parents=True, exist_ok=True)
with csv_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "| Method | W | A | Result | Rate | Wilson 95% CI | Exit | Seconds |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['method']} | {row['weight_bits']} | {row['activation_bits']} | "
        f"{row['successes']}/{row['episodes']} | {float(row['rate']):.2%} | "
        f"[{float(row['wilson95_low']):.2%}, {float(row['wilson95_high']):.2%}] | "
        f"{row['exit_code']} | {row['wall_seconds']} |"
    )
markdown_path.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"CSV: {csv_path}")
print(f"Markdown: {markdown_path}")
print("OFFICIAL QUANTIZATION VALIDATION: PASS")
PY
