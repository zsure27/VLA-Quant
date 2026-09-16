#!/usr/bin/env bash
# Diagnose which LLM stage merits PEFT. PART=A: 0-7/8-15; PART=B: 16-23/24-31.
set -Eeuo pipefail
ROOT=/root/autodl-tmp/qvla-repro
REPO=/root/VLA-Quant
OVERLAY=$ROOT/overlays/language-stage-rescue
BASE=$ROOT/artifacts/awq-spatial-20260912-163735-1136
VALIDATION=$ROOT/artifacts/awq-validation-20260913-132827-1162
PART=${PART:-A}
case "$PART" in
  A) names=(stage-00-07 stage-08-15); layer_sets=(0,1,2,3,4,5,6,7 8,9,10,11,12,13,14,15) ;;
  B) names=(stage-16-23 stage-24-31); layer_sets=(16,17,18,19,20,21,22,23 24,25,26,27,28,29,30,31) ;;
  *) printf 'PART must be A or B\n' >&2; exit 2 ;;
esac
OUT=$ROOT/artifacts/awq-language-stage-rescue-${PART,,}-$(date +%Y%m%d-%H%M%S)-$$
source /root/miniconda3/bin/activate /root/miniconda3/envs/qvla-oft
export PYTHONPATH="$OVERLAY/diagnostics:$REPO:$REPO/diagnostics:$ROOT/src/QVLA/openvla-oft:$ROOT/src/LIBERO"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=7 WANDB_MODE=disabled
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4
test -f "$OVERLAY/diagnostics/probe.py"
test -f "$OVERLAY/diagnostics/awq_interventions.py"
test -f "$OVERLAY/diagnostics/smoothing_selection.py"
test -s "$BASE/profiles/w2.pt"
test -s "$BASE/profiles/w4.pt"
test -s "$VALIDATION/w2-no-clip/metrics.json"
test ! -e "$OUT"
mkdir -p "$OUT"
printf '%s\n' "$OUT" > "$ROOT/artifacts/LATEST_LANGUAGE_STAGE_RESCUE_$PART.txt"
sha256sum "$OVERLAY/diagnostics/probe.py" "$OVERLAY/diagnostics/awq_interventions.py" "$OVERLAY/diagnostics/smoothing_selection.py" > "$OUT/OVERLAY_SHA256SUMS.txt"
git -C "$REPO" rev-parse HEAD > "$OUT/base-commit.txt"
git -C "$REPO" status --short > "$OUT/base-status.txt"
common=(--checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial"
  --samples-dir "$VALIDATION/samples" --official-root "$ROOT/src/official-quantization"
  --targets-file "$REPO/configs/qvla-connected-422.txt" --num-samples 32 --offset 0
  --seed 7 --attention-layers 7,15,23,31 --teacher-dir "$VALIDATION/teacher"
  --mode awq --weight-bits 2 --activation-bits 16 --weight-scope language
  --awq-profile "$BASE/profiles/w2.pt" --awq-disable-clip all
  --awq-w4-profile "$BASE/profiles/w4.pt")
run() {
  local name=$1 layers=$2
  printf '%q ' python "$OVERLAY/diagnostics/probe.py" "${common[@]}" \
    --output "$OUT/$name" --awq-w4-layers "$layers" > "$OUT/$name.command.txt"
  printf '\n' >> "$OUT/$name.command.txt"
  python "$OVERLAY/diagnostics/probe.py" "${common[@]}" \
    --output "$OUT/$name" --awq-w4-layers "$layers" 2>&1 | tee "$OUT/$name.console.log"
}
run "${names[0]}" "${layer_sets[0]}"
run "${names[1]}" "${layer_sets[1]}"
python - "$VALIDATION/w2-no-clip/metrics.json" "$OUT" "${names[@]}" <<'PY'
import json, statistics, sys
from pathlib import Path
baseline_path, root = Path(sys.argv[1]), Path(sys.argv[2])
rows = {"w2-no-clip": json.loads(baseline_path.read_text())}
for name in sys.argv[3:]:
    rows[name] = json.loads((root / name / "metrics.json").read_text())
if any(len(value) != 32 for value in rows.values()):
    raise RuntimeError("Expected 32 matched validation samples per candidate")
names = [{r["sample"] for r in value} for value in rows.values()]
if any(value != names[0] for value in names[1:]):
    raise RuntimeError("Validation sample mismatch")
base = {r["sample"]: r for r in rows["w2-no-clip"]}
summary = {}
for name, value in rows.items():
    errors = [r["normalized_action"]["mse"] for r in value]
    summary[name] = {"mean_action_mse": statistics.mean(errors),
                     "median_action_mse": statistics.median(errors),
                     "gripper_disagreement_steps": round(sum(r["raw_gripper_disagreement"] * 8 for r in value))}
    if name != "w2-no-clip":
        current = {r["sample"]: r for r in value}
        summary[name]["improved_samples"] = sum(current[k]["normalized_action"]["mse"] < base[k]["normalized_action"]["mse"] for k in base)
        summary[name]["worsened_samples"] = sum(current[k]["normalized_action"]["mse"] > base[k]["normalized_action"]["mse"] for k in base)
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(root / "complete.json").write_text(json.dumps({"status": "COMPLETE", "scope": "offline stage localization; no closed-loop or PEFT claim"}, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
(cd "$OUT" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS.txt)
printf 'COMPLETE %s\n' "$OUT"
