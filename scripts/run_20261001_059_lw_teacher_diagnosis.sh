#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/autodl-tmp/qvla-repro
REPO=/root/autodl-tmp/VLA-Quant-lora-expand-20260930
SESSION=$ROOT/backups/experiments/p2-shared-peft/20260930-059-language-w2-all
SOURCE=$SESSION/eval-first50trace/LW
OUT=$SESSION/teacher-on-lw-first50trace
OFT=$ROOT/overlays/awq-p0-stage-20260923/oft
test -s "$SOURCE/on-policy-events.jsonl"
test "$(cat "$SOURCE/exit-code.txt")" = 0
test ! -e "$OUT"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
export PYTHONPATH="$REPO:$OFT:$ROOT/src/LIBERO"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 WANDB_MODE=disabled MUJOCO_GL=egl
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4 TOKENIZERS_PARALLELISM=false
sha256sum "$REPO/qvla/query_same_observation_teacher.py" "$REPO/qvla/on_policy_capture.py" \
  "$SOURCE/on-policy-events.jsonl" "$SOURCE/policy-queries.jsonl" \
  > "$SESSION/teacher-on-lw-first50trace-input-sha256.txt"
/root/miniconda3/envs/qvla-oft/bin/python -u "$REPO/qvla/query_same_observation_teacher.py" \
  --checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial" \
  --student-run "$SOURCE" --output "$OUT" --seed 0
cp "$SESSION/teacher-on-lw-first50trace-input-sha256.txt" "$OUT/CONTRACT_SHA256SUMS.txt"
printf '{"status":"COMPLETE","role":"LW_student_visited_development_reset20_24","training_use":false,"holdout_touched":false}\n' > "$OUT/complete.json"
(cd "$OUT" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS.txt)
