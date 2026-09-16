#!/usr/bin/env bash
# Recheck two controls on the exact eight cached-teacher inputs. No A4 or training.
set -Eeuo pipefail
ROOT=/root/autodl-tmp/qvla-repro
REPO=/root/VLA-Quant
OVERLAY=$ROOT/overlays/language-family-rescue
OLD=$ROOT/artifacts/controls-20260911-234754-1133
OUT=$ROOT/artifacts/sq-controls-recheck-$(date +%Y%m%d-%H%M%S)-$$
CASES=${CASES:-base}
case "$CASES" in
 base) names=(primary-only fused-only); thresholds=(1e-4 1e-4) ;;
 language-negative) names=(no-language no-language-oracle); thresholds=(1e-4 1e-10) ;;
 *) printf 'Unknown CASES\n' >&2; exit 2 ;;
esac
for name in probe.py awq_interventions.py smoothing_selection.py; do
  test -s "$OVERLAY/diagnostics/$name"
done
test -s "$OLD/teacher/manifest.json"
test -s "$OLD/profiles/sq/calibration.pt"
test -s "$ROOT/calib/action-space-balanced/libero-512/sample-0064.npz"
test ! -e "$OUT"
mkdir -p "$OUT"
printf '%s\n' "$OUT" > "$ROOT/artifacts/LATEST_SQ_VISION_BRANCH.txt"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/exit-code.txt"' EXIT
source /root/miniconda3/bin/activate /root/miniconda3/envs/qvla-oft
export PYTHONPATH="$OVERLAY/diagnostics:$REPO:$REPO/diagnostics:$ROOT/src/QVLA/openvla-oft:$ROOT/src/LIBERO"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=7 WANDB_MODE=disabled
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4
git -C "$REPO" rev-parse HEAD > "$OUT/base-commit.txt"
sha256sum "$OVERLAY"/diagnostics/{probe.py,awq_interventions.py,smoothing_selection.py} > "$OUT/OVERLAY_SHA256SUMS.txt"
common=(--checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial"
  --samples-dir "$ROOT/calib/action-space-balanced/libero-512"
  --official-root "$ROOT/src/official-quantization" --targets-file "$REPO/configs/qvla-connected-422.txt"
  --num-samples 8 --offset 64 --seed 7 --attention-layers 7,15,23,31
  --teacher-dir "$OLD/teacher" --weight-bits 16 --activation-bits 16)
run() {
  local name=$1 mode=$2
  shift 2
  printf '%q ' python "$OVERLAY/diagnostics/probe.py" "${common[@]}" \
    --output "$OUT/$name" --mode "$mode" "$@" > "$OUT/$name.command.txt"
  printf '\n' >> "$OUT/$name.command.txt"
  python "$OVERLAY/diagnostics/probe.py" "${common[@]}" \
    --output "$OUT/$name" --mode "$mode" "$@" 2>&1 | tee "$OUT/$name.console.log"
}
if [[ "$CASES" == base ]]; then
 run primary-only smoothquant --profile-dir "$OLD/profiles/sq" --smoothing-selection only-primary-vision
 run fused-only smoothquant --profile-dir "$OLD/profiles/sq" --smoothing-selection only-fused-vision
else
 run no-language smoothquant --profile-dir "$OLD/profiles/sq" --smoothing-selection no-language
 run no-language-oracle smoothquant --profile-dir "$OLD/profiles/sq" --smoothing-selection no-language --oracle-projector
fi
python - "$OUT" "${names[0]}" "${thresholds[0]}" "${names[1]}" "${thresholds[1]}" <<'PY'
import json, statistics, sys
from pathlib import Path
root = Path(sys.argv[1])
summary = {}
for name, threshold in ((sys.argv[2], float(sys.argv[3])), (sys.argv[4], float(sys.argv[5]))):
    records = json.loads((root / name / "metrics.json").read_text())
    errors = [r["normalized_action"]["mse"] for r in records]
    if len(errors) != 8:
        raise RuntimeError("Eight paired control inputs required")
    summary[name] = {"frames": 8, "mean_action_mse": statistics.mean(errors),
        "max_action_mse": max(errors), "gripper_disagreement_steps": round(sum(r["raw_gripper_disagreement"]*8 for r in records)),
        "engineering_threshold": threshold, "within_threshold": max(errors) <= threshold}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(root / "complete.json").write_text(json.dumps({"status": "MEASUREMENT_COMPLETE",
    "tested_controls_within_threshold": all(r["within_threshold"] for r in summary.values()),
    "note": "Completion does not mean the SQ numerical gate passed. Eight old development frames, not final or rollout."}, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
(cd "$OUT" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS.txt)
printf 'CONTROL MEASUREMENTS COMPLETE %s\n' "$OUT"
