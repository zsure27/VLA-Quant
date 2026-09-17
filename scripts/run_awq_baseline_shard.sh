#!/usr/bin/env bash
set -Eeuo pipefail
OFFSET=${1:?offset}; COUNT=${2:?count}
[[ "$OFFSET" =~ ^[0-9]+$ && "$COUNT" =~ ^[0-9]+$ ]] || exit 2
(( COUNT>0 && OFFSET+COUNT<=50 )) || exit 2
ROOT=/root/autodl-tmp/qvla-repro
OVERLAY=$ROOT/overlays/baseline-shards
BASE=$ROOT/artifacts/awq-spatial-20260912-163735-1136
OUT=$ROOT/eval/awq-baseline-shard-$(printf '%02d' "$OFFSET")-$(printf '%02d' "$((OFFSET+COUNT-1))")-$(date +%Y%m%d-%H%M%S)-$$
mkdir -p "$OUT"
printf '%s\n' "$OUT" > "$ROOT/eval/LATEST_BASELINE_SHARD.txt"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/exit-code.txt"' EXIT
source /root/miniconda3/bin/activate /root/miniconda3/envs/qvla-oft
export PYTHONPATH="$OVERLAY:$OVERLAY/oft:$ROOT/src/QVLA/openvla-oft:$ROOT/src/LIBERO"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 WANDB_MODE=disabled MUJOCO_GL=egl
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4
git -C /root/VLA-Quant rev-parse HEAD > "$OUT/base-commit.txt"
cp "$OVERLAY/qvla/run_eval_official_quant.py" "$OUT/evaluator-source.py"
cp "$OVERLAY/oft/experiments/robot/libero/run_libero_eval.py" "$OUT/oft-evaluator-source.py"
sha256sum "$OVERLAY/qvla/run_eval_official_quant.py" "$OVERLAY/oft/experiments/robot/libero/run_libero_eval.py" "$BASE/profiles/w4.pt" > "$OUT/CONTRACT_SHA256SUMS.txt"
common=(--method awq --weight-bits 4 --activation-bits 16
 --pretrained_checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial"
 --profile "$BASE/profiles/w4.pt" --official-root "$ROOT/src/official-quantization"
 --task_suite_name libero_spatial --num_trials_per_task "$COUNT" --initial-state-offset "$OFFSET"
 --libero_root "$ROOT/src/LIBERO" --seed 0 --env-seed 0 --seed-protocol paired --trace-actions)
for name in bf16 w4; do
 scope=all; [[ "$name" == bf16 ]] && scope=none
 mkdir -p "$OUT/$name"
 printf '%q ' python "$OVERLAY/qvla/run_eval_official_quant.py" "${common[@]}" --awq-scope "$scope" --local_log_dir "$OUT/$name" > "$OUT/$name/command.txt"
 printf '\n' >> "$OUT/$name/command.txt"
 python "$OVERLAY/qvla/run_eval_official_quant.py" "${common[@]}" --awq-scope "$scope" --local_log_dir "$OUT/$name" 2>&1 | tee "$OUT/$name/console.log"
 printf '0\n' > "$OUT/$name/exit-code.txt"
done
printf '{"status":"EXECUTION_COMPLETE","note":"Two finite paired shards; BF16 bypass and full AWQ W4, fixed profile; not training or packed kernels."}\n' > "$OUT/complete.json"
