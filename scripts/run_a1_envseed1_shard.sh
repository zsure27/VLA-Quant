#!/usr/bin/env bash
set -Eeuo pipefail

CONFIG=${1:?configuration required}
OFFSET=${2:-5}
COUNT=${3:-5}
[[ "$OFFSET" =~ ^[0-9]+$ && "$COUNT" =~ ^[0-9]+$ ]] || exit 2
(( OFFSET >= 5 && COUNT > 0 && OFFSET + COUNT <= 50 )) || exit 2
case "$CONFIG" in C0|C1|C2|C3|BF16) ;; *) exit 2 ;; esac

ROOT=/root/autodl-tmp/qvla-repro
REPO=/root/autodl-tmp/VLA-Quant-p2c-20260925
OFT=$ROOT/overlays/awq-p0-stage-20260923/oft
BASE=$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w2.pt
W4=$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w4.pt
G64=$ROOT/artifacts/awq-primary-group-20260913-213127-3342/profiles/w2-g64.pt
C1_STATE=$ROOT/backups/experiments/p2-shared-peft/20260926-107-p2-robust-e2e-distill-data80/exact12l-response-svd-r8-smoothl1-data80-e2e1000-offline32/recovery_adapter_state.pt
C3_STATE=$ROOT/backups/experiments/p2-shared-peft/20260929-107-p25-student-state80-distill/exact12l-response-svd-r8-smoothl1-studentstate80-e2e1000/e2e_adapter_state.pt
OUT=$ROOT/eval/a1-110-envseed1-${OFFSET}-$((OFFSET+COUNT-1))/$CONFIG
test ! -e "$OUT/complete.json"
mkdir -p "$OUT"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/exit-code.txt"' EXIT
for path in "$REPO/qvla/run_eval_official_quant.py" "$REPO/qvla/recovery_lora.py" "$OFT/experiments/robot/libero/run_libero_eval.py" "$BASE" "$W4" "$G64" "$C1_STATE" "$C3_STATE"; do test -s "$path"; done
export PYTHONPATH="$REPO:$OFT:$ROOT/src/LIBERO"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 WANDB_MODE=disabled MUJOCO_GL=egl
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4 TOKENIZERS_PARALLELISM=false
sha256sum "$REPO/qvla/run_eval_official_quant.py" "$REPO/qvla/recovery_lora.py" \
  "$OFT/experiments/robot/libero/run_libero_eval.py" "$BASE" "$W4" "$G64" \
  "$C1_STATE" "$C3_STATE" > "$OUT/CONTRACT_SHA256SUMS.txt"

common=(/root/miniconda3/envs/qvla-oft/bin/python -u "$REPO/qvla/run_eval_official_quant.py"
  --method awq --weight-bits 2 --activation-bits 16
  --pretrained_checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial"
  --profile "$BASE" --awq-primary-group64-profile "$G64" --awq-w4-profile "$W4"
  --official-root "$ROOT/src/official-quantization" --task_suite_name libero_spatial
  --num_trials_per_task "$COUNT" --initial-state-offset "$OFFSET" --libero_root "$ROOT/src/LIBERO"
  --seed 0 --env-seed 1 --seed-protocol paired --trace-actions
  --awq-candidate w2-attention-primary-g64-stage-w4)
layers12=8,9,10,11,12,13,14,15,20,21,22,23
layers14=8,9,10,11,12,13,14,15,18,19,20,21,22,23
case "$CONFIG" in
  C0) extra=(--awq-scope all --awq-w4-layers "$layers12") ;;
  C1) extra=(--awq-scope all --awq-w4-layers "$layers12" --awq-recovery-lora-state "$C1_STATE") ;;
  C2) extra=(--awq-scope all --awq-w4-layers "$layers14") ;;
  C3) extra=(--awq-scope all --awq-w4-layers "$layers12" --awq-recovery-lora-state "$C3_STATE") ;;
  BF16) extra=(--awq-scope none) ;;
esac
command=("${common[@]}" "${extra[@]}" --local_log_dir "$OUT")
printf '%q ' "${command[@]}" > "$OUT/command.txt"
printf '\n' >> "$OUT/command.txt"
"${command[@]}" > "$OUT/console.log" 2>&1
printf '{"status":"COMPLETE","config":"%s","env_seed":1,"offset":%s,"count":%s}\n' "$CONFIG" "$OFFSET" "$COUNT" > "$OUT/complete.json"
