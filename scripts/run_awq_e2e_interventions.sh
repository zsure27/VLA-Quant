#!/usr/bin/env bash
set -Eeuo pipefail
# 每组独立进程，视觉保持 BF16；不修改原始 profile。
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export ROOT=${ROOT:-/root/autodl-tmp/qvla-repro}
source "$HERE/scripts/activate_oft.sh"
export PYTHONPATH="$HERE:$HERE/diagnostics:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=7 WANDB_MODE=disabled
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4
export BASE=${BASE:-$ROOT/artifacts/awq-spatial-20260912-163735-1136}
export OUT=${OUT:-$ROOT/artifacts/awq-e2e-interventions-$(date +%Y%m%d-%H%M%S)-$$}
test ! -e "$OUT" || { echo "拒绝覆盖：$OUT"; exit 1; }
mkdir -p "$OUT"
echo "结果目录：$OUT"
python "$HERE/scripts/check_runtime.py" | tee "$OUT/runtime.json"
python "$HERE/diagnostics/gates.py" split --samples "$BASE/samples" \
  --calibration-count 32 --offset 32 --count 32 | tee "$OUT/split.json"
common=(--checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial"
  --samples-dir "$BASE/samples" --official-root "$ROOT/src/official-quantization"
  --targets-file "$HERE/configs/qvla-connected-422.txt" --num-samples 32 --offset 32
  --seed 7 --attention-layers 7,15,23,31 --teacher-dir "$BASE/teacher")
run() {
  local name=$1; shift
  python "$HERE/diagnostics/probe.py" "${common[@]}" --output "$OUT/$name" "$@" \
    2>&1 | tee "$OUT/$name.log"
}
run repeat --mode repeat
python "$HERE/diagnostics/gates.py" control --metrics "$OUT/repeat/metrics.json" --threshold 1e-10
awq=(--mode awq --awq-profile "$BASE/profiles/w2.pt" --weight-bits 2 --activation-bits 16 --weight-scope language)
run baseline "${awq[@]}"
python - <<'PY'
import json, os
from pathlib import Path
base, out = Path(os.environ["BASE"]), Path(os.environ["OUT"])
old = {r["sample"]: r for r in json.loads((base / "awq-w2-language/metrics.json").read_text())}
new = {r["sample"]: r for r in json.loads((out / "baseline/metrics.json").read_text())}
if set(old) != set(new) or len(new) != 32:
    raise RuntimeError("基线样本集合不符")
for name, row in new.items():
    if abs(row["normalized_action"]["mse"] - old[name]["normalized_action"]["mse"]) > 1e-8:
        raise RuntimeError("语言 W2 基线动作误差未复现：" + name)
    if row["raw_gripper_disagreement"] != old[name]["raw_gripper_disagreement"]:
        raise RuntimeError("语言 W2 夹爪指令未复现：" + name)
old_meta = json.loads((base / "awq-w2-language/manifest.json").read_text())
new_meta = json.loads((out / "baseline/manifest.json").read_text())
if old_meta["profiles"] != new_meta["profiles"]:
    raise RuntimeError("W2 profile 发生变化")
print("E2E_BASELINE_REPLAY: PASS")
PY
run no-clip-all "${awq[@]}" --awq-disable-clip all
run no-clip-attention "${awq[@]}" --awq-disable-clip attention
run no-clip-mlp "${awq[@]}" --awq-disable-clip mlp
run w4-blocks-10-12 "${awq[@]}" --awq-w4-layers 10,11,12 --awq-w4-profile "$BASE/profiles/w4.pt"
python - <<'PY'
import json, os, statistics
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
root = Path(os.environ["OUT"])
names = ("repeat", "baseline", "no-clip-all", "no-clip-attention", "no-clip-mlp", "w4-blocks-10-12")
summary = {}
for name in names:
    rows = json.loads((root / name / "metrics.json").read_text())
    if len(rows) != 32:
        raise RuntimeError("样本数不完整：" + name)
    errors = [r["normalized_action"]["mse"] for r in rows]
    onset = {}
    for r in rows:
        onset[r["sample"]] = next((i for i in range(32) if
            (r["features"][f"language_model.model.layers.{i}@0/action_readout"]["relative_mse"] or 0) > 1), None)
    summary[name] = {"mean_action_mse": statistics.mean(errors),
        "median_action_mse": statistics.median(errors), "max_action_mse": max(errors),
        "gripper_disagreement_steps": round(sum(r["raw_gripper_disagreement"] * 8 for r in rows)),
        "first_relative_mse_over_1": onset,
        "per_sample_action_mse": {r["sample"]: r["normalized_action"]["mse"] for r in rows}}
with (root / "summary.json").open("x") as f:
    json.dump(summary, f, indent=2, allow_nan=False)
with (root / "complete.json").open("x") as f:
    json.dump({"status": "COMPLETE", "note": "离线干预完成，不代表成功率通过"}, f)
output = Path(str(root) + "-review.zip")
with ZipFile(output, "x", ZIP_DEFLATED) as z:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in (".json", ".log"):
            z.write(p, str(p.relative_to(root)))
print("AWQ_E2E_INTERVENTIONS: COMPLETE")
for name, values in summary.items():
    print(name, "平均动作MSE=", values["mean_action_mse"], "夹爪分歧=", values["gripper_disagreement_steps"], "/256")
print("请下载并上传：", output)
PY
