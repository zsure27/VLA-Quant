#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/autodl-tmp/qvla-repro
REPO=/root/autodl-tmp/VLA-Quant-lora-expand-20260930
OFT_ROOT=$ROOT/overlays/awq-p0-stage-20260923/oft
SESSION=$ROOT/backups/experiments/p2-shared-peft/20260930-059-language-w2-all
CALIBRATION=$ROOT/backups/experiments/p1-data-contract/20260926-107-p1-data-contract/peft-train-calibration80
STUDENT=$ROOT/backups/experiments/p2-shared-peft/20260929-107-p25-student-state80-distill/student-state80
CHECKPOINT=$ROOT/models/openvla-7b-oft-finetuned-libero-spatial
BASE=$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w2.pt
W4=$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w4.pt
G64=$ROOT/artifacts/awq-primary-group-20260913-213127-3342/profiles/w2-g64.pt
C3=$ROOT/backups/p25studentstate80-20260929-040513-852d350/candidate-adapter.pt
LAYERS=0,1,2,3,4,5,6,7,16,17,18,19,24,25,26,27,28,29,30,31
W4_LAYERS=8,9,10,11,12,13,14,15,20,21,22,23
PY=/root/miniconda3/envs/qvla-oft/bin/python

export PYTHONPATH="$REPO/diagnostics:$REPO:$OFT_ROOT:$ROOT/src/LIBERO"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=7 WANDB_MODE=disabled MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4 TOKENIZERS_PARALLELISM=false

hash_file() { sha256sum "$1"; }
assert_idle_gpu() {
  local pids
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')
  test -z "$pids" || { echo "GPU already has compute PIDs: $pids" >&2; exit 23; }
}

train() {
  local steps=$1 out_name
  case "$steps" in
    10) out_name=language-w2-all-r8-training-smoke10 ;;
    1000) out_name=language-w2-all-r8-e2e1000 ;;
    *) echo "only preregistered 10-step smoke or 1000-step training allowed" >&2; exit 2 ;;
  esac
  local out="$SESSION/$out_name"
  test ! -e "$out"
  assert_idle_gpu
  for path in "$REPO/diagnostics/probe.py" "$REPO/diagnostics/low_rank_recovery.py" \
      "$CHECKPOINT/config.json" "$BASE" "$W4" "$G64" "$STUDENT/manifest.json" \
      "$CALIBRATION/manifest.json" "$ROOT/artifacts/awq-validation-20260913-132827-1162/teacher/manifest.json"; do
    test -s "$path" || { echo "missing preflight material: $path" >&2; exit 4; }
  done
  mkdir -p "$out"
  printf '%s\n' "$steps" > "$out/steps-preregistered.txt"
  {
    hash_file "$REPO/diagnostics/probe.py"
    hash_file "$REPO/diagnostics/low_rank_recovery.py"
    hash_file "$REPO/scripts/verify_20260930_059_lora_training_smoke.py"
    hash_file "$CHECKPOINT/config.json"
    hash_file "$BASE"; hash_file "$W4"; hash_file "$G64"; hash_file "$C3"
    hash_file "$STUDENT/manifest.json"; hash_file "$CALIBRATION/manifest.json"
  } > "$out/CONTRACT_SHA256SUMS.txt"
  local args=(
    --checkpoint "$CHECKPOINT" --samples-dir "$ROOT/artifacts/awq-validation-20260913-132827-1162/samples"
    --official-root "$ROOT/src/official-quantization" --targets-file "$REPO/configs/qvla-connected-422.txt"
    --num-samples 32 --offset 0 --seed 7 --attention-layers 7,15,23,31
    --teacher-dir "$ROOT/artifacts/awq-validation-20260913-132827-1162/teacher"
    --mode awq --weight-bits 2 --activation-bits 16 --weight-scope all
    --awq-profile "$BASE" --awq-disable-clip attention
    --awq-w4-layers "$W4_LAYERS" --awq-w4-profile "$W4"
    --awq-vision-bits 2 --awq-vision-branch all --awq-primary-group64-profile "$G64"
    --awq-exact-attention-visual-stage
    --awq-residual-layers "$LAYERS" --awq-residual-rank 8
    --awq-residual-calibration-dir "$CALIBRATION" --awq-residual-calibration-count 80
    --awq-residual-token-scope action --awq-residual-response-svd
    --awq-e2e-distill-steps "$steps" --awq-e2e-distill-learning-rate 0.0001
    --awq-e2e-distill-loss smooth-l1 --awq-e2e-distill-smooth-l1-beta 0.1
    --awq-e2e-distill-samples-dir "$STUDENT" --awq-e2e-distill-samples-count 80
    --output "$out"
  )
  printf '%q ' "$PY" -u "$REPO/diagnostics/probe.py" "${args[@]}" > "$out/command.txt"
  printf '\n' >> "$out/command.txt"
  set +e
  "$PY" -u "$REPO/diagnostics/probe.py" "${args[@]}" > "$out/console.log" 2>&1
  local rc=$?
  set -e
  printf '%s\n' "$rc" > "$out/exit-code.txt"
  if [[ "$steps" == 10 && "$rc" == 0 ]]; then
    "$PY" "$REPO/scripts/verify_20260930_059_lora_training_smoke.py" --output "$out" > "$out/structural-smoke.json"
  fi
  (cd "$out" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS.txt)
  return "$rc"
}

evaluate() {
  local config=$1 offset=$2 count=$3 shard=$4 state="" scope=all bits=2 profile="$BASE" candidate=w2-attention-primary-g64-stage-w4 extra=()
  [[ "$config" =~ ^(C0|C3|LW|BF16|W4)$ ]] || exit 2
  [[ "$shard" =~ ^(micro|first50|remaining250)$ ]] || exit 2
  case "$shard:$offset:$count" in
    micro:20:1|first50:20:5|remaining250:25:25) ;;
    *) echo "evaluation range is outside the frozen shard plan" >&2; exit 2 ;;
  esac
  case "$config" in
    C0) ;;
    C3) state=$C3; extra+=(--awq-recovery-lora-state "$state") ;;
    LW) state="$SESSION/language-w2-all-r8-e2e1000/e2e_adapter_state.pt"; extra+=(--awq-recovery-lora-state "$state") ;;
    BF16) bits=4; profile=$W4; scope=none; candidate=profile ;;
    W4) bits=4; profile=$W4; scope=all; candidate=profile ;;
  esac
  local out="$SESSION/eval-$shard/$config"
  test ! -e "$out"
  assert_idle_gpu
  mkdir -p "$out"
  local common=(
    --method awq --weight-bits "$bits" --activation-bits 16
    --pretrained_checkpoint "$CHECKPOINT" --profile "$profile"
    --official-root "$ROOT/src/official-quantization" --task_suite_name libero_spatial
    --num_trials_per_task "$count" --initial-state-offset "$offset" --libero_root "$ROOT/src/LIBERO"
    --seed 0 --env-seed 1 --seed-protocol paired --trace-actions --awq-scope "$scope"
    --awq-candidate "$candidate" --local_log_dir "$out"
  )
  if [[ "$config" == C0 || "$config" == C3 || "$config" == LW ]]; then
    common+=(--awq-primary-group64-profile "$G64" --awq-w4-profile "$W4" --awq-w4-layers "$W4_LAYERS")
  fi
  if [[ "$shard" == micro ]]; then common+=(--trace-observations); fi
  printf '%q ' "$PY" -u "$REPO/qvla/run_eval_official_quant.py" "${common[@]}" "${extra[@]}" > "$out/command.txt"
  printf '\n' >> "$out/command.txt"
  {
    hash_file "$REPO/qvla/run_eval_official_quant.py"
    hash_file "$REPO/diagnostics/probe.py"
    hash_file "$CHECKPOINT/config.json"; hash_file "$BASE"; hash_file "$G64"; hash_file "$W4"
    [[ -z "$state" ]] || hash_file "$state"
  } > "$out/CONTRACT_SHA256SUMS.txt"
  set +e
  "$PY" -u "$REPO/qvla/run_eval_official_quant.py" "${common[@]}" "${extra[@]}" > "$out/console.log" 2>&1
  local rc=$?
  set -e
  printf '%s\n' "$rc" > "$out/exit-code.txt"
  (cd "$out" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS.txt)
  return "$rc"
}

case "${1:-}" in
  train) [[ $# == 2 ]] || exit 2; train "$2" ;;
  eval) [[ $# == 5 ]] || exit 2; evaluate "$2" "$3" "$4" "$5" ;;
  *) echo "usage: $0 train {10|1000} | eval {C0|C3|LW|BF16|W4} {20} {1|5|25} {micro|first50|remaining250}" >&2; exit 2 ;;
esac
