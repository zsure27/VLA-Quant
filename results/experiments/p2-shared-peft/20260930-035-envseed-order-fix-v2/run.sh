#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/qvla-repro
PLAN=$ROOT/control/a1-035-envseed-order-fix-smoke-v2-20260930
OUT=$ROOT/eval/a1-035-envseed-order-fix-smoke-v2-20260930
PATCH=$ROOT/overlays/a1-envseed-order-fix-20260930/oft
P2C=/root/autodl-tmp/VLA-Quant-p2c-20260925
LIBERO=$ROOT/src/LIBERO
MODEL=$ROOT/models/openvla-7b-oft-finetuned-libero-spatial
PROFILE=$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w2.pt
EVAL=$P2C/qvla/run_eval_official_quant.py
HELPER=$PATCH/experiments/robot/libero/run_libero_eval.py
PROBE=$ROOT/diagnostics/a1-envseed-order-fix-v2-20260930/probe.py
PY=/root/miniconda3/envs/qvla-oft/bin/python

if [ -e "$OUT" ] || [ -e "$PLAN/status.json" ]; then
  echo 'Refusing to overwrite existing output/status' >&2
  exit 90
fi
mkdir -p "$PLAN" "$OUT"
status() {
  local state="$1" revision="$2" code="$3"
  cat > "$PLAN/status.json.tmp" <<EOF
{"plan_id":"a1-035-envseed-order-fix-smoke-v2-20260930","stage":"ENVSEED_RESET_MASKING_NO_POLICY_SMOKE","status":"$state","revision":$revision,"exit_code":$code,"updated_utc":"$(date -u +%Y-%m-%dT%H:%M:%SZ)","next_stage":"ANALYZE_GATE","auto_continue":false}
EOF
  mv "$PLAN/status.json.tmp" "$PLAN/status.json"
}
trap 'rc=$?; status FAILED 2 "$rc"; exit "$rc"' ERR
status RUNNING 1 null
export MUJOCO_GL=egl TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="$P2C:$PATCH:$LIBERO${PYTHONPATH:+:$PYTHONPATH}"

sha256sum "$EVAL" "$HELPER" "$PROBE" \
  "$P2C/qvla/reproducibility.py" "$PATCH/experiments/robot/libero/libero_utils.py" \
  "$PATCH/experiments/robot/openvla_utils.py" "$PATCH/experiments/robot/robot_utils.py" \
  "$LIBERO/libero/libero/envs/env_wrapper.py" "$LIBERO/libero/libero/envs/bddl_base_domain.py" \
  "$MODEL/config.json" "$MODEL/dataset_statistics.json" \
  "$ROOT/artifacts/a1-014-preflight-20260930/fresh-reset-pilot"/task-*.npy \
  "$PROFILE" > "$OUT/material_sha256_before.txt"
"$PY" -m py_compile "$HELPER" "$PROBE"
"$PY" - <<'PY' > "$OUT/import_assertion.txt"
import inspect
from pathlib import Path
import experiments.robot.libero.run_libero_eval as helper
path = Path(inspect.getsourcefile(helper)).resolve()
expected = Path('/root/autodl-tmp/qvla-repro/overlays/a1-envseed-order-fix-20260930/oft/experiments/robot/libero/run_libero_eval.py')
assert path == expected, (path, expected)
source = inspect.getsource(helper.run_task)
block = source.split('if cfg.seed_protocol == "paired":', 1)[1].split('import hashlib', 1)[0]
assert block.index('seed_all(model_seed') < block.index('env.seed(environment_seed)')
print(f'OFT_HELPER={path}')
print('SEED_ORDER=model_then_environment:PASS')
PY
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader > "$OUT/gpu_before.csv"
"$PY" -u "$PROBE" --output "$OUT/envseed_probe.json" --model-dir "$MODEL" --reset 5 > "$OUT/probe_console.log" 2>&1
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader > "$OUT/gpu_after.csv"
sha256sum "$EVAL" "$HELPER" "$PROBE" \
  "$P2C/qvla/reproducibility.py" "$PATCH/experiments/robot/libero/libero_utils.py" \
  "$PATCH/experiments/robot/openvla_utils.py" "$PATCH/experiments/robot/robot_utils.py" \
  "$LIBERO/libero/libero/envs/env_wrapper.py" "$LIBERO/libero/libero/envs/bddl_base_domain.py" \
  "$MODEL/config.json" "$MODEL/dataset_statistics.json" \
  "$ROOT/artifacts/a1-014-preflight-20260930/fresh-reset-pilot"/task-*.npy \
  "$PROFILE" > "$OUT/material_sha256_after.txt"
cmp -s "$OUT/material_sha256_before.txt" "$OUT/material_sha256_after.txt"
status COMPLETE 2 0
trap - ERR
