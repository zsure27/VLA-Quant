#!/usr/bin/env bash
# Two family tests on exactly the existing 32 validation inputs, no training.
set -Eeuo pipefail
ROOT=/root/autodl-tmp/qvla-repro
REPO=/root/VLA-Quant
OVERLAY=$ROOT/overlays/language-family-rescue
BASE=$ROOT/artifacts/awq-spatial-20260912-163735-1136
VALIDATION=$ROOT/artifacts/awq-validation-20260913-132827-1162
OUT=$ROOT/artifacts/awq-family-precision-$(date +%Y%m%d-%H%M%S)-$$
for name in probe.py awq_interventions.py smoothing_selection.py; do test -s "$OVERLAY/diagnostics/$name"; done
test -s "$VALIDATION/teacher/manifest.json"
mkdir -p "$OUT"
printf '%s\n' "$OUT" > "$ROOT/artifacts/LATEST_STAGE_FIXED_COORDINATES.txt"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/exit-code.txt"' EXIT
source /root/miniconda3/bin/activate /root/miniconda3/envs/qvla-oft
export PYTHONPATH="$OVERLAY/diagnostics:$REPO:$REPO/diagnostics:$ROOT/src/QVLA/openvla-oft:$ROOT/src/LIBERO"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=7 WANDB_MODE=disabled
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4
git -C "$REPO" rev-parse HEAD > "$OUT/base-commit.txt"
sha256sum "$OVERLAY"/diagnostics/{probe.py,awq_interventions.py,smoothing_selection.py} > "$OUT/OVERLAY_SHA256SUMS.txt"
common=(--checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial"
 --samples-dir "$VALIDATION/samples" --official-root "$ROOT/src/official-quantization"
 --targets-file "$REPO/configs/qvla-connected-422.txt" --num-samples 32 --offset 0
 --seed 7 --attention-layers 7,15,23,31 --teacher-dir "$VALIDATION/teacher"
 --mode awq --weight-bits 2 --activation-bits 16 --weight-scope language
 --awq-profile "$BASE/profiles/w2.pt" --awq-disable-clip all
 --awq-rescue-layers 8,9,10,11,12,13,14,15)
for family in all; do
 printf '%q ' python "$OVERLAY/diagnostics/probe.py" "${common[@]}" --awq-rescue-family "$family" --output "$OUT/$family" > "$OUT/$family.command.txt"
 printf '\n' >> "$OUT/$family.command.txt"
 python "$OVERLAY/diagnostics/probe.py" "${common[@]}" --awq-rescue-family "$family" --output "$OUT/$family" 2>&1 | tee "$OUT/$family.console.log"
done
python - "$OUT" <<'PY'
import json, statistics, sys
from pathlib import Path
root = Path(sys.argv[1])
summary = {}
for family in ("all",):
 records = json.loads((root / family / "metrics.json").read_text())
 if len(records) != 32: raise RuntimeError("32 paired development frames required")
 summary[family] = {"mean_action_mse": statistics.mean(r["normalized_action"]["mse"] for r in records),
  "gripper_disagreement_steps": round(sum(r["raw_gripper_disagreement"]*8 for r in records))}
(root / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
(root / "complete.json").write_text(json.dumps({"status": "MEASUREMENT_COMPLETE",
 "note": "Original W2 coordinates, selected family four-bit with no extra clip, no W4-profile optimization, no PEFT or rollout."}, indent=2)+"\n")
print(json.dumps(summary, indent=2))
PY
(cd "$OUT" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS.txt)
printf 'FAMILY PRECISION COMPLETE %s\n' "$OUT"
