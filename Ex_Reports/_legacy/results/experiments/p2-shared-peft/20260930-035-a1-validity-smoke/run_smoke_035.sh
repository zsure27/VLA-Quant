#!/usr/bin/env bash
set -u

ROOT=/root/autodl-tmp/qvla-repro
ART=$ROOT/artifacts/a1-014-preflight-20260930
OUT=$ROOT/eval/a1-035-fresh-reset-bf16-validity-20260930
PLAN=$ROOT/control/a1-035-fresh-reset-bf16-validity-20260930
EVAL=/root/autodl-tmp/VLA-Quant-p2c-20260925/qvla/run_eval_official_quant.py
HELPER=$ROOT/overlays/awq-p0-stage-20260923/oft/experiments/robot/libero/run_libero_eval.py
PROFILE=$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w2.pt
MODEL=$ROOT/models/openvla-7b-oft-finetuned-libero-spatial
PY=/root/miniconda3/envs/qvla-oft/bin/python

if [ -e "$OUT" ] || [ -e "$PLAN/status.json" ]; then
  echo "Refusing to overwrite existing run output or status" >&2
  exit 90
fi
mkdir -p "$PLAN" "$OUT"
status() {
  local state="$1" code="$2"
  local revision=1
  if [ "$state" != "RUNNING" ]; then revision=2; fi
  cat > "$PLAN/status.json.tmp" <<EOF
{"plan_id":"a1-035-fresh-reset-bf16-validity-20260930","stage":"BF16_VALIDITY_10_EPISODES","status":"$state","revision":$revision,"expected_episodes":10,"exit_code":$code,"updated_utc":"$(date -u +%Y-%m-%dT%H:%M:%SZ)","next_stage":"HOLD_FOR_REVIEW","auto_continue":false}
EOF
  mv "$PLAN/status.json.tmp" "$PLAN/status.json"
}

status RUNNING null
export PYTHONPATH="/root/autodl-tmp/VLA-Quant-p2c-20260925:$ROOT/overlays/awq-p0-stage-20260923/oft:$ROOT/src/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl

{
  sha256sum "$ART"/fresh-reset-pilot/task-*.npy "$EVAL" "$HELPER" "$PROFILE" \
    "$MODEL"/config.json "$MODEL"/lora_adapter/adapter_config.json "$MODEL"/lora_adapter/adapter_model.safetensors
} > "$OUT/material_sha256_before.txt" || { status FAILED 91; exit 91; }

if ! "$PY" -m py_compile "$EVAL" "$HELPER"; then status FAILED 92; exit 92; fi
if ! TF_CPP_MIN_LOG_LEVEL=3 "$PY" - <<'PY' > "$OUT/import_assertion.txt" 2>&1
import inspect
import experiments.robot.libero.run_libero_eval as m
p = inspect.getsourcefile(m)
assert p == "/root/autodl-tmp/qvla-repro/overlays/awq-p0-stage-20260923/oft/experiments/robot/libero/run_libero_eval.py", p
assert "initial_state_offset" in inspect.getsource(m)
print("OFT_HELPER=" + p)
print("OFFSET_FLAG=True")
PY
then status FAILED 93; exit 93; fi

cmd=("$PY" -u "$EVAL"
  --method awq --weight-bits 2 --activation-bits 16
  --pretrained_checkpoint "$MODEL"
  --profile "$PROFILE"
  --official-root "$ROOT/src/official-quantization"
  --task_suite_name libero_spatial --num_trials_per_task 1 --initial-state-offset 0
  --a1-reset-dir "$ART/fresh-reset-pilot"
  --libero_root "$ROOT/src/LIBERO"
  --seed 0 --env-seed 0 --seed-protocol paired
  --trace-actions --trace-observations --awq-scope none
  --local_log_dir "$OUT/BF16")
printf '%q ' "${cmd[@]}" > "$OUT/command.txt"
printf '\n' >> "$OUT/command.txt"

"${cmd[@]}" > "$OUT/console.log" 2>&1
code=$?
printf '%s\n' "$code" > "$OUT/exit_code.txt"
{
  sha256sum "$ART"/fresh-reset-pilot/task-*.npy "$EVAL" "$HELPER" "$PROFILE" \
    "$MODEL"/config.json "$MODEL"/lora_adapter/adapter_config.json "$MODEL"/lora_adapter/adapter_model.safetensors
} > "$OUT/material_sha256_after.txt" || code=94
if ! cmp -s "$OUT/material_sha256_before.txt" "$OUT/material_sha256_after.txt"; then code=95; fi
if [ "$code" -eq 0 ]; then status COMPLETE 0; else status FAILED "$code"; fi
exit "$code"
