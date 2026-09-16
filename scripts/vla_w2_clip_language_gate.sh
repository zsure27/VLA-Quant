#!/usr/bin/env bash
# One paired 10-state W2 language-only diagnostic using the original clipped profile.
set -euo pipefail
ROOT=/root/autodl-tmp/qvla-repro
source /root/miniconda3/bin/activate /root/miniconda3/envs/qvla-oft
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1
cd /root/VLA-Quant
run="$ROOT/eval/w2-language-clip10-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$run"
printf '%s\n' "$run" > "$ROOT/eval/LATEST_W2_CLIP_LANGUAGE.txt"
git rev-parse HEAD > "$run/base-commit.txt"
git diff HEAD -- qvla/run_eval_official_quant.py > "$run/evaluator.patch"
command=(python qvla/run_eval_official_quant.py --method awq --weight-bits 2 --activation-bits 16
  --pretrained_checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial"
  --profile "$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w2.pt"
  --official-root "$ROOT/src/official-quantization" --task_suite_name libero_spatial
  --num_trials_per_task 1 --local_log_dir "$run" --libero_root "$ROOT/src/LIBERO"
  --seed 0 --env-seed 0 --seed-protocol paired --awq-scope language --awq-candidate profile)
printf '%q ' "${command[@]}" > "$run/command.txt"
printf '\n' >> "$run/command.txt"
set +e
timeout --signal=TERM --kill-after=30s 25m "${command[@]}" > "$run/console.log" 2>&1
code=$?
set -e
printf '%s\n' "$code" > "$run/exit-code.txt"
(cd "$run" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS.txt)
if test "$code" -ne 0 || grep -q 'Episode error:' "$run"/EVAL-*.txt; then
  printf 'FAILED code=%s run=%s\n' "$code" "$run"
  exit 1
fi
grep -E 'Total episodes:|Total successes:|Overall success rate:' "$run"/EVAL-*.txt
printf 'COMPLETE %s\n' "$run"
