#!/usr/bin/env bash
set -Eeuo pipefail
OFFSET=${1:?offset required}; COUNT=${2:?count required}
[[ "$OFFSET" =~ ^[0-9]+$ && "$COUNT" =~ ^[0-9]+$ ]] || exit 2
(( COUNT == 10 && OFFSET % 10 == 0 && OFFSET + COUNT <= 50 )) || exit 2
ROOT=${VLA_EXPERIMENT_ROOT:-/root/autodl-tmp/qvla-repro}
REPO=${VLA_REPO:-/root/autodl-tmp/VLA-Quant-p2c-20260925}
OFT_ROOT=${VLA_OFT_ROOT:-$ROOT/overlays/awq-p0-stage-20260923/oft}
END=$((OFFSET + COUNT - 1))
matches=("$ROOT"/eval/p25-paired-headroom-${OFFSET}-${END}-*)
(( ${#matches[@]} == 1 )) || { echo "Expected one completed source shard" >&2; exit 3; }
SOURCE=${matches[0]}
STUDENT=$SOURCE/C1-12L-lora
OUT=$ROOT/backups/experiments/p2-shared-peft/20260928-107-p25-onpolicy-teacher500/teacher-${OFFSET}-${END}
test -s "$SOURCE/complete.json"
test -s "$STUDENT/on-policy-events.jsonl"
test -s "$STUDENT/exit-code.txt"
test "$(cat "$STUDENT/exit-code.txt")" = 0
test ! -e "$OUT"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || exit 4
mkdir -p "$(dirname "$OUT")"
export PYTHONPATH="$REPO:$OFT_ROOT:$ROOT/src/LIBERO"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 WANDB_MODE=disabled MUJOCO_GL=egl
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4 TOKENIZERS_PARALLELISM=false
printf '%s\n' "$SOURCE" > "$(dirname "$OUT")/source-${OFFSET}-${END}.txt"
/root/miniconda3/envs/qvla-oft/bin/python -u "$REPO/qvla/query_same_observation_teacher.py" \
  --checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial" \
  --student-run "$STUDENT" --output "$OUT" --seed 0
sha256sum "$REPO/qvla/query_same_observation_teacher.py" "$REPO/qvla/on_policy_capture.py" \
  "$STUDENT/on-policy-events.jsonl" > "$OUT/CONTRACT_SHA256SUMS.txt"
printf '{"status":"COMPLETE","distribution":"student-visited","holdout_touched":false}\n' > "$OUT/complete.json"
