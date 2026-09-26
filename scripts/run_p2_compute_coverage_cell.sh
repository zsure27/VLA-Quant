#!/usr/bin/env bash
set -Eeuo pipefail

CELL=${1:?usage: run_p2_compute_coverage_cell.sh B-or-C}
case "$CELL" in
  B) TRAJECTORIES=16; STEPS=1000 ;;
  C) TRAJECTORIES=80; STEPS=200 ;;
  *) echo "Only missing matrix cells B and C may be launched; A and D already exist" >&2; exit 2 ;;
esac

ROOT=${VLA_EXPERIMENT_ROOT:-/root/autodl-tmp/qvla-repro}
REPO=${VLA_REPO:-/root/autodl-tmp/VLA-Quant-p2c-20260925}
SESSION=${VLA_SESSION:-20260926-107-p2-compute-coverage}
CALIBRATION_ROOT=$ROOT/backups/experiments/p1-data-contract
if (( TRAJECTORIES == 16 )); then
  CALIBRATION=${VLA_PEFT_CALIBRATION:-$CALIBRATION_ROOT/20260925-107-p1-data-contract/peft-train-calibration16}
else
  CALIBRATION=${VLA_PEFT_CALIBRATION:-$CALIBRATION_ROOT/20260926-107-p1-data-contract/peft-train-calibration80}
fi
OUT_ROOT=$ROOT/backups/experiments/p2-shared-peft/$SESSION
OUT=$OUT_ROOT/cell-${CELL}-traj${TRAJECTORIES}-steps${STEPS}
test ! -e "$OUT"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || exit 3
mkdir -p "$OUT_ROOT"

cat > "$OUT_ROOT/cell-${CELL}-preregistered-protocol.json" <<JSON
{
  "schema_version": "1.0",
  "question": "Does offline generalization come from trajectory diversity or total optimization exposure?",
  "status": "PREREGISTERED_BEFORE_EXECUTION",
  "matrix_cell": "$CELL",
  "trajectory_coverage": $TRAJECTORIES,
  "optimizer_steps": $STEPS,
  "samples_seen": $STEPS,
  "effective_exposures_per_selected_frame": $(python -c "print($STEPS/$TRAJECTORIES)"),
  "fixed": {"backbone":"exact-12L","blocks":[18,19],"rank":8,"initialization":"Response-SVD","loss":"Smooth-L1(beta=0.1)","learning_rate":0.0001,"batch_semantics":"one selected trajectory frame per optimizer step"},
  "primary_gate": "trajectory-level router_dev; historical 32 frames are regression/debug only",
  "closed_loop_planned": false
}
JSON

export PYTHONPATH="$REPO/diagnostics:$REPO:$ROOT/src/QVLA/openvla-oft:$ROOT/src/LIBERO"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=7 WANDB_MODE=disabled
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4
START=$(date +%s)
/root/miniconda3/envs/qvla-oft/bin/python -u "$REPO/diagnostics/probe.py" \
  --checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial" \
  --samples-dir "$ROOT/artifacts/awq-validation-20260913-132827-1162/samples" \
  --official-root "$ROOT/src/official-quantization" --targets-file "$REPO/configs/qvla-connected-422.txt" \
  --num-samples 32 --offset 0 --seed 7 --attention-layers 7,15,23,31 \
  --teacher-dir "$ROOT/artifacts/awq-validation-20260913-132827-1162/teacher" \
  --mode awq --weight-bits 2 --activation-bits 16 --weight-scope all \
  --awq-profile "$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w2.pt" \
  --awq-disable-clip attention --awq-w4-layers 8,9,10,11,12,13,14,15,20,21,22,23 \
  --awq-w4-profile "$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w4.pt" \
  --awq-vision-bits 2 --awq-vision-branch all \
  --awq-primary-group64-profile "$ROOT/artifacts/awq-primary-group-20260913-213127-3342/profiles/w2-g64.pt" \
  --awq-exact-attention-visual-stage --awq-residual-layers 18,19 --awq-residual-rank 8 \
  --awq-residual-calibration-dir "$CALIBRATION" --awq-residual-calibration-count "$TRAJECTORIES" \
  --awq-residual-token-scope action --awq-residual-response-svd \
  --awq-e2e-distill-steps "$STEPS" --awq-e2e-distill-learning-rate 0.0001 \
  --awq-e2e-distill-loss smooth-l1 --awq-e2e-distill-smooth-l1-beta 0.1 --output "$OUT" \
  2>&1 | tee "$OUT_ROOT/cell-${CELL}-console.log"
END=$(date +%s)
python - "$OUT_ROOT/cell-${CELL}-runtime.json" "$START" "$END" <<'PY'
import json,pathlib,sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({"start_unix":int(sys.argv[2]),"end_unix":int(sys.argv[3]),"wall_seconds":int(sys.argv[3])-int(sys.argv[2])},indent=2)+"\n")
PY
(cd "$OUT_ROOT" && find . -type f ! -name SESSION_SHA256SUMS.txt -print0 | sort -z | xargs -0 -r sha256sum > SESSION_SHA256SUMS.txt)

