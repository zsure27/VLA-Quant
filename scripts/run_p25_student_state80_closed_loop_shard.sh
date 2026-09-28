#!/usr/bin/env bash
set -Eeuo pipefail
OFFSET=${1:?offset required}; COUNT=${2:?count required}
[[ "$OFFSET" =~ ^[0-9]+$ && "$COUNT" =~ ^[0-9]+$ ]] || exit 2
(( (OFFSET == 5 && COUNT == 5) || (OFFSET >= 10 && OFFSET % 10 == 0 && COUNT == 10 && OFFSET + COUNT <= 50) )) || exit 2
ROOT=${VLA_EXPERIMENT_ROOT:-/root/autodl-tmp/qvla-repro}
REPO=${VLA_REPO:-/root/autodl-tmp/VLA-Quant-p2c-20260925}
OFT_ROOT=${VLA_OFT_ROOT:-$ROOT/overlays/awq-p0-stage-20260923/oft}
STATE=$ROOT/backups/experiments/p2-shared-peft/20260929-107-p25-student-state80-distill/exact12l-response-svd-r8-smoothl1-studentstate80-e2e1000/e2e_adapter_state.pt
BASE=$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w2.pt
W4=$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w4.pt
G64=$ROOT/artifacts/awq-primary-group-20260913-213127-3342/profiles/w2-g64.pt
SOURCE_OFFSET=$OFFSET; SOURCE_END=$((OFFSET+COUNT-1))
if (( OFFSET == 5 )); then SOURCE_OFFSET=0; SOURCE_END=9; fi
matches=("$ROOT"/eval/p25-paired-headroom-${SOURCE_OFFSET}-${SOURCE_END}-*)
(( ${#matches[@]} == 1 )) || exit 3
REFERENCE=${matches[0]}
test -s "$REFERENCE/paired-episodes.jsonl"
test -s "$REFERENCE/CONTRACT_SHA256SUMS.txt"
test -s "$STATE"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || exit 4
(cd / && sha256sum --status -c "$REFERENCE/CONTRACT_SHA256SUMS.txt")
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=$ROOT/eval/p25-student-state80-${OFFSET}-$((OFFSET+COUNT-1))-$STAMP-$$
test ! -e "$OUT"
mkdir -p "$OUT/C3-12L-student-state"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/exit-code.txt"' EXIT
export PYTHONPATH="$REPO:$OFT_ROOT:$ROOT/src/LIBERO"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 WANDB_MODE=disabled MUJOCO_GL=egl
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4 TOKENIZERS_PARALLELISM=false
printf '%s\n' "$REFERENCE" > "$OUT/frozen-reference-path.txt"
sha256sum "$REPO/qvla/run_eval_official_quant.py" "$REPO/qvla/recovery_lora.py" \
  "$OFT_ROOT/experiments/robot/libero/run_libero_eval.py" "$STATE" "$BASE" "$G64" "$W4" \
  > "$OUT/CONTRACT_SHA256SUMS.txt"
/root/miniconda3/envs/qvla-oft/bin/python -u "$REPO/qvla/run_eval_official_quant.py" \
  --method awq --weight-bits 2 --activation-bits 16 \
  --pretrained_checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial" \
  --profile "$BASE" --awq-primary-group64-profile "$G64" --awq-w4-profile "$W4" \
  --official-root "$ROOT/src/official-quantization" --task_suite_name libero_spatial \
  --num_trials_per_task "$COUNT" --initial-state-offset "$OFFSET" --libero_root "$ROOT/src/LIBERO" \
  --seed 0 --env-seed 0 --seed-protocol paired --trace-actions --awq-scope all \
  --awq-candidate w2-attention-primary-g64-stage-w4 \
  --awq-w4-layers 8,9,10,11,12,13,14,15,20,21,22,23 \
  --awq-recovery-lora-state "$STATE" --local_log_dir "$OUT/C3-12L-student-state" \
  > "$OUT/C3-12L-student-state/console.log" 2>&1
printf '0\n' > "$OUT/C3-12L-student-state/exit-code.txt"
/root/miniconda3/envs/qvla-oft/bin/python "$REPO/scripts/summarize_p25_student_state_shard.py" \
  --reference "$REFERENCE" --candidate "$OUT" --offset "$OFFSET" --count "$COUNT"
printf '{"status":"COMPLETE","classification":"paired_with_frozen_reference","resets":"5-49_only","holdout_touched":false}\n' > "$OUT/complete.json"
