#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/autodl-tmp/qvla-repro
OVERLAY=$ROOT/overlays/awq-p0-stage-20260923
RUNNER=$OVERLAY/scripts/run_awq_p0_stage_shard.sh
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || exit 3
BATCH=$ROOT/eval/awq-p0-stage-noncontiguous-full-batch-$(date +%Y%m%d-%H%M%S)-$$
mkdir -p "$BATCH"; printf '%s\n' "$BATCH" > "$ROOT/eval/LATEST_AWQ_P0_STAGE_NONCONTIGUOUS.txt"
for layers in \
  8,9,10,11,12,13,14,15,20,21,22,23 \
  8,9,10,11,16,17,18,19,20,21,22,23; do
  bash "$RUNNER" "$layers" 5 5
  cat "$ROOT/eval/LATEST_AWQ_P0_STAGE.txt" >> "$BATCH/RUNS.txt"
done
printf '{"status":"EXECUTION_COMPLETE","runs":2,"episodes_per_run":50}\n' > "$BATCH/complete.json"
