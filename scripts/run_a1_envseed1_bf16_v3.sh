#!/usr/bin/env bash
set -Eeuo pipefail
OFFSET=${1:-5}; COUNT=${2:-5}
[[ "$OFFSET" =~ ^[0-9]+$ && "$COUNT" =~ ^[0-9]+$ ]] || exit 2
(( OFFSET >= 5 && COUNT > 0 && OFFSET + COUNT <= 50 )) || exit 2
ROOT=/root/autodl-tmp/qvla-repro
REPO=/root/autodl-tmp/VLA-Quant-p2c-20260925
OFT=$ROOT/overlays/awq-p0-stage-20260923/oft
BASE=$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w2.pt
OUT=$ROOT/eval/a1-110-envseed1-${OFFSET}-$((OFFSET+COUNT-1))/BF16
test ! -e "$OUT/complete.json"
mkdir -p "$OUT"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/exit-code.txt"' EXIT
export PYTHONPATH="$REPO:$OFT:$ROOT/src/LIBERO"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 WANDB_MODE=disabled MUJOCO_GL=egl
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4 TOKENIZERS_PARALLELISM=false
sha256sum "$REPO/qvla/run_eval_official_quant.py" "$OFT/experiments/robot/libero/run_libero_eval.py" "$BASE" > "$OUT/CONTRACT_SHA256SUMS.txt"
command=(/root/miniconda3/envs/qvla-oft/bin/python -u "$REPO/qvla/run_eval_official_quant.py"
  --method awq --weight-bits 2 --activation-bits 16
  --pretrained_checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial"
  --profile "$BASE" --official-root "$ROOT/src/official-quantization"
  --task_suite_name libero_spatial --num_trials_per_task "$COUNT"
  --initial-state-offset "$OFFSET" --libero_root "$ROOT/src/LIBERO"
  --seed 0 --env-seed 1 --seed-protocol paired --trace-actions
  --awq-scope none --local_log_dir "$OUT")
printf '%q ' "${command[@]}" > "$OUT/command.txt"
printf '\n' >> "$OUT/command.txt"
"${command[@]}" > "$OUT/console.log" 2>&1
printf '{"status":"COMPLETE","config":"BF16","env_seed":1,"offset":%s,"count":%s}\n' "$OFFSET" "$COUNT" > "$OUT/complete.json"
