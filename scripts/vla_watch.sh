#!/usr/bin/env bash
# Read-only status view usable in both VS Code Remote-SSH and AutoDL Web Terminal.
set -eu
ROOT=/root/autodl-tmp/qvla-repro
date --iso-8601=seconds
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
ps -eo pid,etime,args | grep -E '[p]ython.*run_eval|[t]imeout.*run_eval|[b]ash.*vla_diagnostic_batch' || true
if test -f "$ROOT/eval/LATEST_DIAGNOSTIC.txt"; then
  batch=$(cat "$ROOT/eval/LATEST_DIAGNOSTIC.txt")
  printf 'Results: %s\n' "$batch"
  for mode in bf16 w4 language vision; do
    if test -f "$batch/$mode/console.log"; then
      printf '\n%s\n' "$mode"
      tail -5 "$batch/$mode/console.log"
    fi
  done
fi
