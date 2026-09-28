#!/usr/bin/env bash
set -Eeuo pipefail
OFFSET=${1:?offset required}; COUNT=${2:?count required}
[[ "$OFFSET" =~ ^[0-9]+$ && "$COUNT" =~ ^[0-9]+$ ]] || exit 2
(( COUNT == 10 && OFFSET % 10 == 0 && OFFSET + COUNT <= 50 )) || exit 2
ROOT=${VLA_EXPERIMENT_ROOT:-/root/autodl-tmp/qvla-repro}
REPO=${VLA_REPO:-/root/autodl-tmp/VLA-Quant-p2c-20260925}
OFT_ROOT=${VLA_OFT_ROOT:-$ROOT/src/QVLA/openvla-oft}
END=$((OFFSET + COUNT - 1))
matches=("$ROOT"/eval/p25-paired-headroom-${OFFSET}-${END}-*)
(( ${#matches[@]} == 1 )) || exit 3
SOURCE=${matches[0]}/C1-12L-lora
OUT=$ROOT/backups/experiments/p2-shared-peft/20260928-107-p25-query3-no-lora/control-${OFFSET}-${END}
test ! -e "$OUT"
test -s "$SOURCE/on-policy-events.jsonl"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || exit 4
mkdir -p "$OUT"
export PYTHONPATH="$REPO/diagnostics:$REPO:$OFT_ROOT:$ROOT/src/LIBERO"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=7 WANDB_MODE=disabled
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4
/root/miniconda3/envs/qvla-oft/bin/python "$REPO/scripts/prepare_p25_query3_samples.py" \
  --student-run "$SOURCE" --output "$OUT/samples"
common=(--checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial"
  --samples-dir "$OUT/samples" --official-root "$ROOT/src/official-quantization"
  --targets-file "$REPO/configs/qvla-connected-422.txt" --offset 0 --seed 7
  --num-samples 100 --attention-layers '' --action-only)
/root/miniconda3/envs/qvla-oft/bin/python -u "$REPO/diagnostics/probe.py" "${common[@]}" \
  --mode teacher --output "$OUT/bf16-teacher" > "$OUT/bf16-console.log" 2>&1
/root/miniconda3/envs/qvla-oft/bin/python -u "$REPO/diagnostics/probe.py" "${common[@]}" \
  --mode awq --weight-bits 2 --activation-bits 16 --weight-scope all \
  --awq-profile "$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w2.pt" \
  --awq-disable-clip attention --awq-w4-layers 8,9,10,11,12,13,14,15,20,21,22,23 \
  --awq-w4-profile "$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w4.pt" \
  --awq-vision-bits 2 --awq-vision-branch all \
  --awq-primary-group64-profile "$ROOT/artifacts/awq-primary-group-20260913-213127-3342/profiles/w2-g64.pt" \
  --awq-exact-attention-visual-stage --teacher-dir "$OUT/bf16-teacher" \
  --output "$OUT/exact12l-no-lora" > "$OUT/exact12l-console.log" 2>&1
sha256sum "$REPO/diagnostics/probe.py" "$REPO/scripts/prepare_p25_query3_samples.py" \
  "$SOURCE/on-policy-events.jsonl" > "$OUT/CONTRACT_SHA256SUMS.txt"
printf '{"status":"COMPLETE","queries":100,"selection":"student query 3","holdout_touched":false}\n' > "$OUT/complete.json"
