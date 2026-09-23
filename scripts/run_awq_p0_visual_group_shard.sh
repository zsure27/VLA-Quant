#!/usr/bin/env bash
# P0 attribution: fixed G64 attention-only language intervention with DINO G64 / SigLIP G128.
set -Eeuo pipefail
RECIPE=${1:?dino-g64-siglip-g128|dino-g128-siglip-g64|visual-g128}
OFFSET=${2:?official initial state offset}
COUNT=${3:?states per task}
[[ "$OFFSET" =~ ^[0-9]+$ && "$COUNT" =~ ^[0-9]+$ ]] || exit 2
(( COUNT > 0 && OFFSET + COUNT <= 50 )) || exit 2

ROOT=/root/autodl-tmp/qvla-repro
OVERLAY=$ROOT/overlays/awq-p0-20260923
BASE=$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w2.pt
G64=$ROOT/artifacts/awq-primary-group-20260913-213127-3342/profiles/w2-g64.pt
case "$RECIPE" in
  dino-g64-siglip-g128) CANDIDATE=w2-attention-no-clip-primary-g64 ;;
  dino-g128-siglip-g64) CANDIDATE=w2-attention-no-clip-dino-g128-siglip-g64 ;;
  visual-g128) CANDIDATE=w2-attention-no-clip-visual-g128 ;;
  *) exit 2 ;;
esac
MODE=full-w2-language-g64-attention-$RECIPE

test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || {
  echo 'GPU already occupied; refusing concurrent policy runs'
  exit 3
}
OUT=$ROOT/eval/awq-p0-$MODE-$(printf '%02d' "$OFFSET")-$(printf '%02d' "$((OFFSET+COUNT-1))")-$(date +%Y%m%d-%H%M%S)-$$
mkdir -p "$OUT/$MODE"
printf '%s\n' "$OUT" > "$ROOT/eval/LATEST_AWQ_P0.txt"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/exit-code.txt"' EXIT

source /root/miniconda3/bin/activate /root/miniconda3/envs/qvla-oft
export PYTHONPATH="$OVERLAY:$OVERLAY/oft:$ROOT/src/QVLA/openvla-oft:$ROOT/src/LIBERO"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 WANDB_MODE=disabled MUJOCO_GL=egl
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4

git -C /root/VLA-Quant rev-parse HEAD > "$OUT/base-commit.txt"
cp "$OVERLAY/qvla/run_eval_official_quant.py" "$OUT/evaluator-source.py"
cp "$OVERLAY/diagnostics/awq_interventions.py" "$OUT/awq-interventions-source.py"
cp "$OVERLAY/oft/experiments/robot/libero/run_libero_eval.py" "$OUT/oft-evaluator-source.py"
sha256sum "$OVERLAY/qvla/run_eval_official_quant.py" \
  "$OVERLAY/diagnostics/awq_interventions.py" \
  "$OVERLAY/oft/experiments/robot/libero/run_libero_eval.py" \
  "$BASE" "$G64" > "$OUT/CONTRACT_SHA256SUMS.txt"

command=(python "$OVERLAY/qvla/run_eval_official_quant.py"
  --method awq --weight-bits 2 --activation-bits 16
  --pretrained_checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial"
  --profile "$BASE" --awq-primary-group64-profile "$G64"
  --official-root "$ROOT/src/official-quantization"
  --task_suite_name libero_spatial --num_trials_per_task "$COUNT" --initial-state-offset "$OFFSET"
  --libero_root "$ROOT/src/LIBERO" --seed 0 --env-seed 0 --seed-protocol paired --trace-actions
  --awq-scope all --awq-candidate "$CANDIDATE"
  --local_log_dir "$OUT/$MODE")
printf '%q ' "${command[@]}" > "$OUT/$MODE/command.txt"
printf '\n' >> "$OUT/$MODE/command.txt"
"${command[@]}" 2>&1 | tee "$OUT/$MODE/console.log"
printf '0\n' > "$OUT/$MODE/exit-code.txt"
printf '{"status":"EXECUTION_COMPLETE","note":"P0 AWQ visual-group attribution; language is fixed G64 attention-only no-clip; finite fake-quant rollout."}\n' > "$OUT/complete.json"
