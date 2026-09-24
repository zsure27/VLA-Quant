#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/autodl-tmp/qvla-repro
OVERLAY=$ROOT/overlays/awq-p0-stage-20260923
SOURCE_RUNNER=$OVERLAY/scripts/run_awq_p0_stage_shard.sh
OFFSET=${1:-10}
COUNT=${2:-5}

[[ "$OFFSET" =~ ^[0-9]+$ && "$COUNT" =~ ^[1-9][0-9]*$ ]]
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || exit 3

BATCH=$ROOT/eval/awq-p0-stage-generalization-pair-$(date +%Y%m%d-%H%M%S)-$$
mkdir -p "$BATCH"
printf '%s\n' "$BATCH" > "$ROOT/eval/LATEST_AWQ_P0_STAGE_GENERALIZATION.txt"

# Freeze the exact runner used by both paired candidates.
RUNNER=$BATCH/run_awq_p0_stage_shard_frozen.sh
cp "$SOURCE_RUNNER" "$RUNNER"
chmod +x "$RUNNER"

# Reference 16-block protection, followed by the best 12-block pruning found
# on states 5-9. Both use the same paired state and model seed protocol.
for layers in \
  8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 \
  8,9,10,11,12,13,14,15,20,21,22,23
do
  bash "$RUNNER" "$layers" "$OFFSET" "$COUNT"
  cat "$ROOT/eval/LATEST_AWQ_P0_STAGE.txt" >> "$BATCH/RUNS.txt"
done

printf '{"status":"EXECUTION_COMPLETE","runs":2,"episodes_per_run":%d,"offset":%d}\n' "$((10*COUNT))" "$OFFSET" > "$BATCH/complete.json"
