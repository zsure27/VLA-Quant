#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/autodl-tmp/qvla-repro
OVERLAY=$ROOT/overlays/awq-p0-stage-20260923
RUNNER=$OVERLAY/scripts/run_awq_p0_stage_shard.sh
REFERENCE=${1:?existing reference result directory}
LAYERS=${2:?comma-separated candidate W4 blocks}
OFFSET=${3:?initial-state offset}
COUNT=${4:?initial-state count}

test -d "$REFERENCE"
test -f "$REFERENCE/complete.json"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || exit 3

BATCH=$ROOT/eval/awq-p0-stage-candidate-pair-$(date +%Y%m%d-%H%M%S)-$$
mkdir -p "$BATCH"
printf '%s\n' "$BATCH" > "$ROOT/eval/LATEST_AWQ_P0_STAGE_CANDIDATE_PAIR.txt"
printf '%s\n' "$REFERENCE" > "$BATCH/RUNS.txt"
bash "$RUNNER" "$LAYERS" "$OFFSET" "$COUNT"
cat "$ROOT/eval/LATEST_AWQ_P0_STAGE.txt" >> "$BATCH/RUNS.txt"
printf '{"status":"EXECUTION_COMPLETE","runs":2,"episodes_per_run":%d,"offset":%d,"candidate_layers":"%s","reference_reused":true}\n' \
  "$((10*COUNT))" "$OFFSET" "$LAYERS" > "$BATCH/complete.json"
