#!/usr/bin/env bash
# Two finite paired ten-state diagnostics, language only; no recovery training.
set -Eeuo pipefail
ROOT=/root/autodl-tmp/qvla-repro
OVERLAY=$ROOT/overlays/closed-loop-fixed-coordinates
BASE=$ROOT/artifacts/awq-spatial-20260912-163735-1136
OUT=$ROOT/eval/language-stage-rescue10-$(date +%Y%m%d-%H%M%S)-$$
test -s "$OVERLAY/qvla/run_eval_official_quant.py"
test -s "$OVERLAY/diagnostics/awq_interventions.py"
test ! -e "$OUT"
mkdir -p "$OUT"
printf '%s\n' "$OUT" > "$ROOT/eval/LATEST_STAGE_FIXED_ROLLOUT10.txt"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/exit-code.txt"' EXIT
source /root/miniconda3/bin/activate /root/miniconda3/envs/qvla-oft
export PYTHONPATH="$OVERLAY:$ROOT/src/QVLA/openvla-oft:$ROOT/src/LIBERO"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 WANDB_MODE=disabled
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4
export MUJOCO_GL=egl
git -C /root/VLA-Quant rev-parse HEAD > "$OUT/base-commit.txt"
sha256sum "$OVERLAY/qvla/run_eval_official_quant.py" "$OVERLAY/diagnostics/awq_interventions.py" > "$OUT/OVERLAY_SHA256SUMS.txt"
cp "$OVERLAY/qvla/run_eval_official_quant.py" "$OUT/evaluator-source.py"
cp "$OVERLAY/diagnostics/awq_interventions.py" "$OUT/intervention-source.py"
common=(--method awq --weight-bits 2 --activation-bits 16
 --pretrained_checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial"
 --profile "$BASE/profiles/w2.pt" 
 --official-root "$ROOT/src/official-quantization" --task_suite_name libero_spatial
 --num_trials_per_task 1 --libero_root "$ROOT/src/LIBERO"
 --seed 0 --env-seed 0 --seed-protocol paired --awq-scope language
 --awq-candidate w2-fixed-coordinates-stage-w4 --trace-actions)
run() {
 local name=$1 layers=$2
 mkdir -p "$OUT/$name"
 printf '%q ' python "$OVERLAY/qvla/run_eval_official_quant.py" "${common[@]}" \
  --awq-w4-layers "$layers" --local_log_dir "$OUT/$name" > "$OUT/$name/command.txt"
 printf '\n' >> "$OUT/$name/command.txt"
 set +e
 python "$OVERLAY/qvla/run_eval_official_quant.py" "${common[@]}" \
  --awq-w4-layers "$layers" --local_log_dir "$OUT/$name" 2>&1 | tee "$OUT/$name/console.log"
 local rc=${PIPESTATUS[0]}
 set -e
 printf '%s\n' "$rc" > "$OUT/$name/exit-code.txt"
 if (( rc != 0 )); then return "$rc"; fi
}
run stage-08-15 8,9,10,11,12,13,14,15

printf '{"status":"EXECUTION_COMPLETE","note":"Inspect episode manifests and success; fake mixed precision, no PEFT."}\n' > "$OUT/complete.json"
printf 'ROLLOUT BATCH COMPLETE %s\n' "$OUT"
