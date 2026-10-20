#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/autodl-tmp/qvla-repro
PLAN=$ROOT/control/a1-035-envseed1-first50-tail40-20260930
OUT=$ROOT/eval/a1-035-envseed1-first50-tail40-20260930
P2C=/root/autodl-tmp/VLA-Quant-p2c-20260925
PATCH=$ROOT/overlays/a1-envseed-order-fix-20260930/oft
LIBERO=$ROOT/src/LIBERO
MODEL=$ROOT/models/openvla-7b-oft-finetuned-libero-spatial
BASE=$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w2.pt
W4=$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w4.pt
G64=$ROOT/artifacts/awq-primary-group-20260913-213127-3342/profiles/w2-g64.pt
C3=$ROOT/backups/experiments/p2-shared-peft/20260929-107-p25-student-state80-distill/exact12l-response-svd-r8-smoothl1-studentstate80-e2e1000/e2e_adapter_state.pt
EVAL=$P2C/qvla/run_eval_official_quant.py
HELPER=$PATCH/experiments/robot/libero/run_libero_eval.py
PY=/root/miniconda3/envs/qvla-oft/bin/python
if [ -e "$OUT" ] || [ -e "$PLAN/status.json" ]; then echo 'Refusing overwrite' >&2; exit 90; fi
mkdir -p "$OUT" "$PLAN"
status() {
  local state="$1" revision="$2" code="$3" case_name="$4"
  local continue_flag=false
  if [ "$state" = RUNNING ]; then continue_flag=true; fi
  cat > "$PLAN/status.json.tmp" <<EOF
{"plan_id":"a1-035-envseed1-first50-tail40-20260930","status":"$state","revision":$revision,"exit_code":$code,"current_case":"$case_name","next_stage":"ANALYZE_HARD_GATE","auto_continue":$continue_flag,"updated_utc":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF
  mv "$PLAN/status.json.tmp" "$PLAN/status.json"
}
trap 'rc=$?; status FAILED 3 "$rc" "${active_case:-PREFLIGHT}"; exit "$rc"' ERR
status RUNNING 1 null C0
export PYTHONPATH="$P2C:$PATCH:$LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 WANDB_MODE=disabled MUJOCO_GL=egl
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4 TOKENIZERS_PARALLELISM=false
sha256sum "$EVAL" "$P2C/qvla/recovery_lora.py" "$HELPER" "$BASE" "$W4" "$G64" "$C3" \
  "$MODEL/config.json" "$MODEL/dataset_statistics.json" > "$OUT/material_sha256_before.txt"
test "$(sha256sum "$EVAL" | cut -d' ' -f1)" = b4efd426d521e0e44438df48b4e8aba39280d9e4c3160e0ac8ca1913d87207b6
test "$(sha256sum "$HELPER" | cut -d' ' -f1)" = a4061279eebe82944f048bebfc9e85fa85acf0d7a9cf5d6ce18aea1102694fe2
test "$(sha256sum "$BASE" | cut -d' ' -f1)" = 4fe1d2aa9a4e89fbaa5f9eb358ac6526d899195f774378b476742b801c7f4ccc
test "$(sha256sum "$G64" | cut -d' ' -f1)" = 947a849114a402978ae60995a652cd3212ded8d66f6ee7e0f3eeb3484712b3b8
test "$(sha256sum "$W4" | cut -d' ' -f1)" = ea3faca6130c88bd38fbe20be28eafc742d66605d339f8d9acfedb8da4703ea4
test "$(sha256sum "$C3" | cut -d' ' -f1)" = 67cd6a6d4ec75d9173ce95b5f8f21c99b6562bfea58ea80947aa5640a335710a
"$PY" - <<'PY' > "$OUT/import_assertion.txt"
import inspect
import experiments.robot.libero.run_libero_eval as helper
path=inspect.getsourcefile(helper)
assert path == '/root/autodl-tmp/qvla-repro/overlays/a1-envseed-order-fix-20260930/oft/experiments/robot/libero/run_libero_eval.py', path
print('OFT_HELPER='+path)
PY
run_case() {
  local case_name="$1" case_dir="$OUT/$1"
  active_case="$case_name"
  mkdir -p "$case_dir"
  local extra=()
  if [ "$case_name" = C3 ]; then extra=(--awq-recovery-lora-state "$C3"); fi
  local cmd=("$PY" -u "$EVAL" --method awq --weight-bits 2 --activation-bits 16
    --pretrained_checkpoint "$MODEL" --profile "$BASE" --awq-primary-group64-profile "$G64"
    --awq-w4-profile "$W4" --official-root "$ROOT/src/official-quantization"
    --task_suite_name libero_spatial --num_trials_per_task 4 --initial-state-offset 6
    --libero_root "$LIBERO" --seed 0 --env-seed 1 --seed-protocol paired
    --trace-actions --trace-observations --awq-candidate w2-attention-primary-g64-stage-w4
    --awq-scope all --awq-w4-layers 8,9,10,11,12,13,14,15,20,21,22,23
    "${extra[@]}" --local_log_dir "$case_dir")
  printf '%q ' "${cmd[@]}" > "$case_dir/command.txt"; printf '\n' >> "$case_dir/command.txt"
  "${cmd[@]}" > "$case_dir/console.log" 2>&1
  printf '0\n' > "$case_dir/exit-code.txt"
  printf '{"status":"COMPLETE","config":"%s","env_seed":1,"offset":6,"count":4}\n' "$case_name" > "$case_dir/complete.json"
}
run_case C0
status RUNNING 2 null C3
run_case C3
sha256sum "$EVAL" "$P2C/qvla/recovery_lora.py" "$HELPER" "$BASE" "$W4" "$G64" "$C3" \
  "$MODEL/config.json" "$MODEL/dataset_statistics.json" > "$OUT/material_sha256_after.txt"
cmp -s "$OUT/material_sha256_before.txt" "$OUT/material_sha256_after.txt"
"$PY" "$PLAN/summarize_a1_envseed1.py" "$OUT" --offset 6 --count 4 > "$OUT/paired-summary-console.json"
status COMPLETE 3 0 NONE
trap - ERR
