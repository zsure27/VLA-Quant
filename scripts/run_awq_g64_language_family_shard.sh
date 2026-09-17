#!/usr/bin/env bash
# Finite follow-up AWQ ablation. Launch only after the prioritized W4 baseline.
set -Eeuo pipefail
FAMILY=${1:?attention|mlp}; OFFSET=${2:?official initial state offset}; COUNT=${3:?states per task}
[[ "$FAMILY" == attention || "$FAMILY" == mlp ]] || exit 2
[[ "$OFFSET" =~ ^[0-9]+$ && "$COUNT" =~ ^[0-9]+$ ]] || exit 2
(( COUNT>0 && OFFSET+COUNT<=50 )) || exit 2
MODE=language-w2-g64-no-clip-$FAMILY
ROOT=/root/autodl-tmp/qvla-repro
OVERLAY=$ROOT/overlays/baseline-shards
BASE=$ROOT/artifacts/awq-primary-group-20260913-213127-3342
PROFILE=$ROOT/artifacts/awq-language-g64-family-no-clip-derived/w2-g64-language-no-clip-$FAMILY.pt
bits=2; candidate=profile; scope=language; extra=()
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || { echo 'GPU already occupied; refusing concurrent policy runs'; exit 3; }
OUT=$ROOT/eval/awq-scope-shard-$MODE-$(printf '%02d' "$OFFSET")-$(printf '%02d' "$((OFFSET+COUNT-1))")-$(date +%Y%m%d-%H%M%S)-$$
mkdir -p "$OUT/$MODE"
printf '%s\n' "$OUT" > "$ROOT/eval/LATEST_AWQ_SCOPE_SHARD.txt"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/exit-code.txt"' EXIT
source /root/miniconda3/bin/activate /root/miniconda3/envs/qvla-oft
export PYTHONPATH="$OVERLAY:$OVERLAY/oft:$ROOT/src/QVLA/openvla-oft:$ROOT/src/LIBERO"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 WANDB_MODE=disabled MUJOCO_GL=egl
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4
git -C /root/VLA-Quant rev-parse HEAD > "$OUT/base-commit.txt"
cp "${PROFILE%.pt}.intervention.json" "$OUT/profile-intervention.json"
cp "$OVERLAY/qvla/run_eval_official_quant.py" "$OUT/evaluator-source.py"
cp "$OVERLAY/oft/experiments/robot/libero/run_libero_eval.py" "$OUT/oft-evaluator-source.py"
sha256sum "$OVERLAY/qvla/run_eval_official_quant.py" "$OVERLAY/oft/experiments/robot/libero/run_libero_eval.py" "$PROFILE" > "$OUT/CONTRACT_SHA256SUMS.txt"

command=(python "$OVERLAY/qvla/run_eval_official_quant.py" --method awq --weight-bits "$bits" --activation-bits 16
 --pretrained_checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial"
 --profile "$PROFILE" --official-root "$ROOT/src/official-quantization"
 --task_suite_name libero_spatial --num_trials_per_task "$COUNT" --initial-state-offset "$OFFSET"
 --libero_root "$ROOT/src/LIBERO" --seed 0 --env-seed 0 --seed-protocol paired --trace-actions
 --awq-scope "$scope" --awq-candidate "$candidate" "${extra[@]}" --local_log_dir "$OUT/$MODE")
printf '%q ' "${command[@]}" > "$OUT/$MODE/command.txt"; printf '\n' >> "$OUT/$MODE/command.txt"
"${command[@]}" 2>&1 | tee "$OUT/$MODE/console.log"
printf '0\n' > "$OUT/$MODE/exit-code.txt"
printf '{"status":"EXECUTION_COMPLETE","note":"Finite AWQ scope ablation; not uniform whole-model W2, training or packed execution."}\n' > "$OUT/complete.json"
