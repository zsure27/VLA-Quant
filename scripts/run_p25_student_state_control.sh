#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/autodl-tmp/qvla-repro
REPO=/root/autodl-tmp/VLA-Quant-p2c-20260925
SOURCE=$ROOT/backups/experiments/p2-shared-peft/20260927-107-p25-on-policy-alignment/student-rollout
OUT=$ROOT/backups/experiments/p2-shared-peft/20260927-107-p25-student-state-control
SAMPLES=$OUT/student-visited-samples
TEACHER=$OUT/bf16-probe
BASELINE=$OUT/exact12l-no-lora
test ! -e "$OUT"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
source /root/miniconda3/bin/activate /root/miniconda3/envs/qvla-oft
export PYTHONPATH="$REPO/diagnostics:$REPO:$ROOT/src/QVLA/openvla-oft:$ROOT/src/LIBERO"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=7 WANDB_MODE=disabled
export TF_NUM_INTEROP_THREADS=2 TF_INTRA_OP_PARALLELISM_THREADS=4
python "$REPO/scripts/prepare_p25_probe_samples.py" --student-run "$SOURCE" --output "$SAMPLES"
common=(--checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial" --samples-dir "$SAMPLES"
  --official-root "$ROOT/src/official-quantization" --targets-file "$REPO/configs/qvla-connected-422.txt"
  --offset 0 --seed 7 --num-samples 140 --attention-layers '' --action-only)
python -u "$REPO/diagnostics/probe.py" "${common[@]}" --mode teacher --output "$TEACHER" > "$OUT/bf16-console.log" 2>&1
python -u "$REPO/diagnostics/probe.py" "${common[@]}" --mode awq --weight-bits 2 --activation-bits 16 --weight-scope all \
  --awq-profile "$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w2.pt" --awq-disable-clip attention \
  --awq-w4-layers 8,9,10,11,12,13,14,15,20,21,22,23 \
  --awq-w4-profile "$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w4.pt" \
  --awq-vision-bits 2 --awq-vision-branch all \
  --awq-primary-group64-profile "$ROOT/artifacts/awq-primary-group-20260913-213127-3342/profiles/w2-g64.pt" \
  --awq-exact-attention-visual-stage --teacher-dir "$TEACHER" --output "$BASELINE" > "$OUT/exact12l-console.log" 2>&1
(cd "$OUT" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS.txt)
printf '{"status":"COMPLETE","student_visited_queries":140,"holdout_touched":false}\n' > "$OUT/complete.json"
