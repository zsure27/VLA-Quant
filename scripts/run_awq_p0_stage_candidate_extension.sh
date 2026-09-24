#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/autodl-tmp/qvla-repro
OVERLAY=$ROOT/overlays/awq-p0-stage-20260923
PAIR=$OVERLAY/scripts/run_awq_p0_stage_candidate_against_existing.sh
LAYERS=8,9,10,11,12,13,14,15,18,19,20,21,22,23
SUPER=$ROOT/eval/awq-p0-stage-candidate-extension-$(date +%Y%m%d-%H%M%S)-$$
mkdir -p "$SUPER"
printf '%s\n' "$SUPER" > "$ROOT/eval/LATEST_AWQ_P0_STAGE_CANDIDATE_EXTENSION.txt"

for item in \
  '30|/root/autodl-tmp/qvla-repro/eval/awq-p0-stage-full-w2-dino64-siglip128-language-g64-attention-stage-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23-30-34-20260924-235937-7257' \
  '35|/root/autodl-tmp/qvla-repro/eval/awq-p0-stage-full-w2-dino64-siglip128-language-g64-attention-stage-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23-35-39-20260925-002108-12925' \
  '40|/root/autodl-tmp/qvla-repro/eval/awq-p0-stage-full-w2-dino64-siglip128-language-g64-attention-stage-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23-40-44-20260925-004229-18584' \
  '45|/root/autodl-tmp/qvla-repro/eval/awq-p0-stage-full-w2-dino64-siglip128-language-g64-attention-stage-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23-45-49-20260925-010357-24228'
do
  offset=${item%%|*}
  reference=${item#*|}
  bash "$PAIR" "$reference" "$LAYERS" "$offset" 5
  cat "$ROOT/eval/LATEST_AWQ_P0_STAGE_CANDIDATE_PAIR.txt" >> "$SUPER/BATCHES.txt"
done

printf '{"status":"EXECUTION_COMPLETE","candidate_layers":"%s","batches":4,"episodes":200}\n' "$LAYERS" > "$SUPER/complete.json"
