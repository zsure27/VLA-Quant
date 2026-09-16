#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/qvla-repro
REPO=/root/VLA-Quant
source /root/miniconda3/bin/activate /root/miniconda3/envs/qvla-oft
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1
cd "$REPO"
batch="$ROOT/eval/diagnostic-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$batch"
printf '%s\n' "$batch" > "$ROOT/eval/LATEST_DIAGNOSTIC.txt"
git rev-parse HEAD > "$batch/base-commit.txt"
git diff -- qvla/run_eval_official_quant.py > "$batch/evaluator.patch"
sha256sum qvla/run_eval_official_quant.py > "$batch/evaluator.sha256"
for mode in bf16 w4 language vision; do
  run="$batch/$mode"
  mkdir -p "$run"
  bits=2
  scope="$mode"
  candidate=w2-no-clip-primary-g64
  extra=(--awq-primary-group64-profile "$ROOT/artifacts/awq-primary-group-20260913-213127-3342/profiles/w2-g64.pt")
  if test "$mode" = bf16; then scope=none; fi
  if test "$mode" = w4; then bits=4; scope=all; candidate=profile; extra=(); fi
  command=(python qvla/run_eval_official_quant.py --method awq --weight-bits "$bits" --activation-bits 16
    --pretrained_checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial"
    --profile "$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w$bits.pt"
    --official-root "$ROOT/src/official-quantization" --task_suite_name libero_spatial
    --num_trials_per_task 1 --local_log_dir "$run" --libero_root "$ROOT/src/LIBERO"
    --seed 0 --env-seed 0 --seed-protocol paired --awq-candidate "$candidate"
    --awq-scope "$scope" --trace-actions "${extra[@]}")
  printf '%q ' "${command[@]}" > "$run/command.txt"
  printf '\n' >> "$run/command.txt"
  printf 'START %s %s\n' "$mode" "$(date --iso-8601=seconds)"
  set +e
  timeout --signal=TERM --kill-after=30s 12m "${command[@]}" > "$run/console.log" 2>&1
  code=$?
  set -e
  printf '%s\n' "$code" > "$run/exit-code.txt"
  (cd "$run" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.txt)
  if test "$code" -ne 0 || grep -q 'Episode error:' "$run"/EVAL-*.txt; then
    printf 'FAILED %s code=%s; stopping batch\n' "$mode" "$code"
    exit 1
  fi
  grep -E 'Total episodes:|Total successes:|Overall success rate:' "$run"/EVAL-*.txt
done
printf 'BATCH_COMPLETE %s\n' "$batch"
