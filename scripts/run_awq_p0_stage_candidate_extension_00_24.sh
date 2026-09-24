#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/autodl-tmp/qvla-repro
PAIR=$ROOT/overlays/awq-p0-stage-20260923/scripts/run_awq_p0_stage_candidate_against_existing.sh
LAYERS=8,9,10,11,12,13,14,15,18,19,20,21,22,23
SUPER=$ROOT/eval/awq-p0-stage-candidate-extension-00-24-$(date +%Y%m%d-%H%M%S)-$$
mkdir -p "$SUPER"
printf '%s\n' "$SUPER" > "$ROOT/eval/LATEST_AWQ_P0_STAGE_CANDIDATE_EXTENSION_00_24.txt"

for item in \
  '0|/root/autodl-tmp/qvla-repro/eval/awq-p0-stage-full-w2-dino64-siglip128-language-g64-attention-stage-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23-00-04-20260925-012513-30715' \
  '5|/root/autodl-tmp/qvla-repro/eval/awq-p0-stage-full-w2-dino64-siglip128-language-g64-attention-stage-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23-05-09-20260923-183044-1649' \
  '10|/root/autodl-tmp/qvla-repro/eval/awq-p0-stage-full-w2-dino64-siglip128-language-g64-attention-stage-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23-10-14-20260924-123019-1516' \
  '15|/root/autodl-tmp/qvla-repro/eval/awq-p0-stage-full-w2-dino64-siglip128-language-g64-attention-stage-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23-15-19-20260924-125246-7292' \
  '20|/root/autodl-tmp/qvla-repro/eval/awq-p0-stage-full-w2-dino64-siglip128-language-g64-attention-stage-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23-20-24-20260924-132013-13065'
do
  offset=${item%%|*}
  reference=${item#*|}
  bash "$PAIR" "$reference" "$LAYERS" "$offset" 5
  cat "$ROOT/eval/LATEST_AWQ_P0_STAGE_CANDIDATE_PAIR.txt" >> "$SUPER/BATCHES.txt"
done

printf '{"status":"EXECUTION_COMPLETE","candidate_layers":"%s","batches":5,"episodes":250}\n' "$LAYERS" > "$SUPER/complete.json"
