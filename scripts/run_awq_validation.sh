#!/usr/bin/env bash
set -Eeuo pipefail
# 固定方案的新轨迹验证；不使用结果选择样本，不重校准。
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export ROOT=${ROOT:-/root/autodl-tmp/qvla-repro}
source "$HERE/scripts/activate_oft.sh"
export PYTHONPATH="$HERE:$HERE/diagnostics:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=7 WANDB_MODE=disabled
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4
export BASE=${BASE:-$ROOT/artifacts/awq-spatial-20260912-163735-1136}
export OUT=${OUT:-$ROOT/artifacts/awq-validation-$(date +%Y%m%d-%H%M%S)-$$}
test ! -e "$OUT" || { echo "拒绝覆盖：$OUT"; exit 1; }
mkdir -p "$OUT"
echo "结果目录：$OUT"
python - <<'PY'
import hashlib, json, os
from pathlib import Path
base = Path(os.environ["BASE"])
for bits, case in ((2, "awq-w2-language"), (4, "awq-w4-all")):
    path = base / f"profiles/w{bits}.pt"
    expected = json.loads((base / case / "manifest.json").read_text())["profiles"][0]["sha256"]
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    if h.hexdigest() != expected:
        raise RuntimeError("固定 profile 已变化：" + str(path))
print("FROZEN_PROFILES: PASS")
PY
python "$HERE/scripts/prepare_spatial_validation.py" \
  --source "$ROOT/calib/action-space-balanced/libero-512" --base "$BASE" \
  --artifacts "$ROOT/artifacts" --output "$OUT/samples" | tee "$OUT/split.log"
python "$HERE/scripts/check_runtime.py" | tee "$OUT/runtime.json"
common=(--checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial"
  --samples-dir "$OUT/samples" --official-root "$ROOT/src/official-quantization"
  --targets-file "$HERE/configs/qvla-connected-422.txt" --num-samples 32 --offset 0
  --seed 7 --attention-layers 7,15,23,31 --teacher-dir "$OUT/teacher")
run() {
  local name=$1; shift
  python "$HERE/diagnostics/probe.py" "${common[@]}" --output "$OUT/$name" "$@" \
    2>&1 | tee "$OUT/$name.log"
}
run teacher --mode teacher
run repeat --mode repeat
python "$HERE/diagnostics/gates.py" control --metrics "$OUT/repeat/metrics.json" --threshold 1e-10
awq=(--mode awq --activation-bits 16 --weight-scope language)
run w2-baseline "${awq[@]}" --weight-bits 2 --awq-profile "$BASE/profiles/w2.pt"
run w2-no-clip "${awq[@]}" --weight-bits 2 --awq-profile "$BASE/profiles/w2.pt" --awq-disable-clip all
run w4-language "${awq[@]}" --weight-bits 4 --awq-profile "$BASE/profiles/w4.pt"
python - <<'PY'
import json, os, statistics
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
root = Path(os.environ["OUT"])
summary = {}
for name in ("repeat", "w2-baseline", "w2-no-clip", "w4-language"):
    rows = json.loads((root / name / "metrics.json").read_text())
    if len(rows) != 32:
        raise RuntimeError("样本数量不符：" + name)
    errors = [r["normalized_action"]["mse"] for r in rows]
    summary[name] = {"mean_action_mse": statistics.mean(errors),
        "median_action_mse": statistics.median(errors), "max_action_mse": max(errors),
        "gripper_disagreement_steps": round(sum(r["raw_gripper_disagreement"]*8 for r in rows)),
        "per_sample_action_mse": {r["sample"]: r["normalized_action"]["mse"] for r in rows}}
base = summary["w2-baseline"]["per_sample_action_mse"]
candidate = summary["w2-no-clip"]["per_sample_action_mse"]
summary["paired"] = {"improved": sum(candidate[n] < base[n] for n in base),
                     "worsened": sum(candidate[n] > base[n] for n in base),
                     "note": "32条新诊断轨迹，不等同闭环成功率"}
with (root / "summary.json").open("x", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
with (root / "complete.json").open("x", encoding="utf-8") as f:
    json.dump({"status": "COMPLETE", "note": "方案固定的新样本验证完成，尚未 rollout"}, f)
output = Path(str(root) + "-review.zip")
with ZipFile(output, "x", ZIP_DEFLATED) as z:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in (".json", ".log"):
            z.write(p, str(p.relative_to(root)))
print("AWQ_VALIDATION: COMPLETE")
for name, row in summary.items():
    if "mean_action_mse" in row:
        print(name, "平均动作MSE=", row["mean_action_mse"], "夹爪分歧=", row["gripper_disagreement_steps"], "/256")
print("请下载并上传：", output)
PY
