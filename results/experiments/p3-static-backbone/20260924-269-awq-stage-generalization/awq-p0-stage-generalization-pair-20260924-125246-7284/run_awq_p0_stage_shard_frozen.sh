#!/usr/bin/env bash
set -Eeuo pipefail
LAYERS=${1:?comma-separated W4 blocks}; OFFSET=${2:?offset}; COUNT=${3:?count}
[[ "$OFFSET" =~ ^[0-9]+$ && "$COUNT" =~ ^[0-9]+$ ]] || exit 2
(( COUNT > 0 && OFFSET + COUNT <= 50 )) || exit 2
ROOT=/root/autodl-tmp/qvla-repro
OVERLAY=$ROOT/overlays/awq-p0-stage-20260923
BASE=$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w2.pt
W4=$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w4.pt
G64=$ROOT/artifacts/awq-primary-group-20260913-213127-3342/profiles/w2-g64.pt
MODE=full-w2-dino64-siglip128-language-g64-attention-stage-${LAYERS//,/-}
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || exit 3
OUT=$ROOT/eval/awq-p0-stage-$MODE-$(printf '%02d' "$OFFSET")-$(printf '%02d' "$((OFFSET+COUNT-1))")-$(date +%Y%m%d-%H%M%S)-$$
mkdir -p "$OUT/$MODE"; printf '%s\n' "$OUT" > "$ROOT/eval/LATEST_AWQ_P0_STAGE.txt"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/exit-code.txt"' EXIT
source /root/miniconda3/bin/activate /root/miniconda3/envs/qvla-oft
export PYTHONPATH="$OVERLAY:$OVERLAY/oft:$ROOT/src/QVLA/openvla-oft:$ROOT/src/LIBERO"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 WANDB_MODE=disabled MUJOCO_GL=egl
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4 TOKENIZERS_PARALLELISM=false
git -C /root/VLA-Quant rev-parse HEAD > "$OUT/base-commit.txt"
cp "$OVERLAY/qvla/run_eval_official_quant.py" "$OUT/evaluator-source.py"
cp "$OVERLAY/diagnostics/awq_interventions.py" "$OUT/awq-interventions-source.py"
cp "$OVERLAY/oft/experiments/robot/libero/run_libero_eval.py" "$OUT/oft-evaluator-source.py"
sha256sum "$OVERLAY/qvla/run_eval_official_quant.py" "$OVERLAY/diagnostics/awq_interventions.py" \
  "$OVERLAY/oft/experiments/robot/libero/run_libero_eval.py" "$BASE" "$G64" "$W4" > "$OUT/CONTRACT_SHA256SUMS.txt"
command=(python "$OVERLAY/qvla/run_eval_official_quant.py" --method awq --weight-bits 2 --activation-bits 16
 --pretrained_checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial"
 --profile "$BASE" --awq-primary-group64-profile "$G64" --awq-w4-profile "$W4" --awq-w4-layers "$LAYERS"
 --official-root "$ROOT/src/official-quantization" --task_suite_name libero_spatial
 --num_trials_per_task "$COUNT" --initial-state-offset "$OFFSET" --libero_root "$ROOT/src/LIBERO"
 --seed 0 --env-seed 0 --seed-protocol paired --trace-actions --awq-scope all
 --awq-candidate w2-attention-primary-g64-stage-w4 --local_log_dir "$OUT/$MODE")
printf '%q ' "${command[@]}" > "$OUT/$MODE/command.txt"; printf '\n' >> "$OUT/$MODE/command.txt"
"${command[@]}" 2>&1 | tee "$OUT/$MODE/console.log"
printf '0\n' > "$OUT/$MODE/exit-code.txt"
printf '{"status":"EXECUTION_COMPLETE","note":"Full-scope P0 stage rescue with DINO64/SigLIP128 and G64 attention-only language remainder."}\n' > "$OUT/complete.json"
