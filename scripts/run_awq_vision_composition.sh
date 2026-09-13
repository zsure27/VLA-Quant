#!/usr/bin/env bash
set -Eeuo pipefail
# 仅回到旧诊断集；新验证集不用于本轮选择方案。
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export ROOT=${ROOT:-/root/autodl-tmp/qvla-repro}
source "$HERE/scripts/activate_oft.sh"
export PYTHONPATH="$HERE:$HERE/diagnostics:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=7 WANDB_MODE=disabled
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4
export BASE=${BASE:-$ROOT/artifacts/awq-spatial-20260912-163735-1136}
export PREVIOUS=${PREVIOUS:-$ROOT/artifacts/awq-e2e-interventions-20260912-173315-1136}
export OUT=${OUT:-$ROOT/artifacts/awq-vision-composition-$(date +%Y%m%d-%H%M%S)-$$}
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
        raise RuntimeError("固定校准参数已变化：" + str(path))
print("FROZEN_PROFILES: PASS")
PY
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
awq=(--mode awq --weight-bits 2 --activation-bits 16 --awq-profile "$BASE/profiles/w2.pt" --awq-disable-clip all)
run vision-bf16 "${awq[@]}" --weight-scope language
python - <<'PY'
import json, os
from pathlib import Path
old = json.loads((Path(os.environ["PREVIOUS"]) / "no-clip-all/metrics.json").read_text())
new = json.loads((Path(os.environ["OUT"]) / "vision-bf16/metrics.json").read_text())
a, b = ({r["sample"]: r for r in rows} for rows in (old, new))
if set(a) != set(b) or len(b) != 32:
    raise RuntimeError("复现样本集合不同")
for name in b:
    if abs(a[name]["normalized_action"]["mse"] - b[name]["normalized_action"]["mse"]) > 1e-8:
        raise RuntimeError("无裁剪语言 W2 参照未复现：" + name)
    if a[name]["raw_gripper_disagreement"] != b[name]["raw_gripper_disagreement"]:
        raise RuntimeError("无裁剪语言 W2 夹爪指令未复现：" + name)
print("LANGUAGE_NO_CLIP_REPLAY: PASS")
PY
run vision-w4 "${awq[@]}" --weight-scope all --awq-vision-bits 4 --awq-vision-profile "$BASE/profiles/w4.pt"
run vision-w2 "${awq[@]}" --weight-scope all --awq-vision-bits 2
python - <<'PY'
import json, os, statistics
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
root = Path(os.environ["OUT"])
summary = {}
for name in ("repeat", "vision-bf16", "vision-w4", "vision-w2"):
    rows = json.loads((root / name / "metrics.json").read_text())
    if len(rows) != 32:
        raise RuntimeError("样本不完整：" + name)
    if name != "repeat":
        scope = json.loads((root / name / "scope.json").read_text())
        if len(scope["weight_targets"]) != (224 if name == "vision-bf16" else 422):
            raise RuntimeError("实际量化范围不符")
        if len(scope["removed_clip_targets"]) != 160 or any(n.startswith("vision_backbone.") for n in scope["removed_clip_targets"]):
            raise RuntimeError("裁剪干预范围不符")
    errors = [r["normalized_action"]["mse"] for r in rows]
    summary[name] = {"mean_action_mse": statistics.mean(errors),
        "median_action_mse": statistics.median(errors), "max_action_mse": max(errors),
        "gripper_disagreement_steps": round(sum(r["raw_gripper_disagreement"] * 8 for r in rows)),
        "per_sample_action_mse": {r["sample"]: r["normalized_action"]["mse"] for r in rows}}
with (root / "summary.json").open("x") as f:
    json.dump(summary, f, indent=2, allow_nan=False)
with (root / "complete.json").open("x") as f:
    json.dump({"status": "COMPLETE", "note": "旧诊断集视觉组合完成，不是 rollout 成功率"}, f)
output = Path(str(root) + "-review.zip")
with ZipFile(output, "x", ZIP_DEFLATED) as z:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in (".json", ".log"):
            z.write(p, str(p.relative_to(root)))
print("AWQ_VISION_COMPOSITION: COMPLETE")
for name, values in summary.items():
    print(name, "平均动作MSE=", values["mean_action_mse"], "夹爪分歧=", values["gripper_disagreement_steps"], "/256")
print("请下载并上传：", output)
PY
