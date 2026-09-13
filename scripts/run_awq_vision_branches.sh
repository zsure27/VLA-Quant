#!/usr/bin/env bash
set -Eeuo pipefail
# 固定语言无裁剪 W2，仅改变被量化的视觉编码器；不改变相机输入。
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export ROOT=${ROOT:-/root/autodl-tmp/qvla-repro}
source "$HERE/scripts/activate_oft.sh"
export PYTHONPATH="$HERE:$HERE/diagnostics:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=7 WANDB_MODE=disabled
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4
export BASE=${BASE:-$ROOT/artifacts/awq-spatial-20260912-163735-1136}
export PREVIOUS=${PREVIOUS:-$ROOT/artifacts/awq-vision-composition-20260913-141517-1162}
export OUT=${OUT:-$ROOT/artifacts/awq-vision-branches-$(date +%Y%m%d-%H%M%S)-$$}
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
awq=(--mode awq --weight-bits 2 --activation-bits 16 --awq-profile "$BASE/profiles/w2.pt" --awq-disable-clip all)
run vision-bf16 "${awq[@]}" --weight-scope language
run vision-w2-all "${awq[@]}" --weight-scope all --awq-vision-bits 2
python - <<'PY'
import json, os
from pathlib import Path
old_root, new_root = Path(os.environ["PREVIOUS"]), Path(os.environ["OUT"])
for old_name, new_name in (("vision-bf16", "vision-bf16"), ("vision-w2", "vision-w2-all")):
    old, new = [json.loads((r / n / "metrics.json").read_text()) for r, n in ((old_root, old_name), (new_root, new_name))]
    a, b = ({v["sample"]: v for v in rows} for rows in (old, new))
    if set(a) != set(b) or len(b) != 32:
        raise RuntimeError("参照样本集合变化")
    for name in b:
        if abs(a[name]["normalized_action"]["mse"] - b[name]["normalized_action"]["mse"]) > 1e-8:
            raise RuntimeError("参照动作误差未复现：" + name)
        if a[name]["raw_gripper_disagreement"] != b[name]["raw_gripper_disagreement"]:
            raise RuntimeError("参照夹爪指令未复现：" + name)
    old_meta = json.loads((old_root / old_name / "manifest.json").read_text())
    new_meta = json.loads((new_root / new_name / "manifest.json").read_text())
    if old_meta["profiles"] != new_meta["profiles"]:
        raise RuntimeError("profile 已变化")
print("VISION_REFERENCES_REPLAY: PASS")
PY
run vision-w2-primary "${awq[@]}" --weight-scope all --awq-vision-bits 2 --awq-vision-branch primary
run vision-w2-fused "${awq[@]}" --weight-scope all --awq-vision-bits 2 --awq-vision-branch fused
python - <<'PY'
import json, os, statistics
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
root = Path(os.environ["OUT"])
summary = {}
for name, count in (("repeat", 0), ("vision-bf16", 224), ("vision-w2-all", 422),
                    ("vision-w2-primary", 317), ("vision-w2-fused", 329)):
    rows = json.loads((root / name / "metrics.json").read_text())
    if len(rows) != 32:
        raise RuntimeError("样本不完整：" + name)
    if count:
        scope = json.loads((root / name / "scope.json").read_text())
        targets = scope["weight_targets"]
        if len(targets) != count or sum(n.startswith("language_model.") for n in targets) != 224:
            raise RuntimeError("量化范围不符：" + name)
        if len(scope["removed_clip_targets"]) != 160 or any(n.startswith("vision_backbone.") for n in scope["removed_clip_targets"]):
            raise RuntimeError("取消裁剪范围错误")
        if name == "vision-w2-primary" and any(n.startswith("vision_backbone.fused_featurizer.") for n in targets):
            raise RuntimeError("主视觉组误量化另一个分支")
        if name == "vision-w2-fused" and any(n.startswith("vision_backbone.featurizer.") for n in targets):
            raise RuntimeError("融合视觉组误量化另一个分支")
    errors = [r["normalized_action"]["mse"] for r in rows]
    summary[name] = {"mean_action_mse": statistics.mean(errors), "median_action_mse": statistics.median(errors),
        "max_action_mse": max(errors), "gripper_disagreement_steps": round(sum(r["raw_gripper_disagreement"]*8 for r in rows)),
        "per_sample_action_mse": {r["sample"]: r["normalized_action"]["mse"] for r in rows}}
with (root / "summary.json").open("x") as f:
    json.dump(summary, f, indent=2, allow_nan=False)
with (root / "complete.json").open("x") as f:
    json.dump({"status": "COMPLETE", "note": "视觉编码器分支诊断完成，不是相机消融或闭环成功率"}, f)
output = Path(str(root) + "-review.zip")
with ZipFile(output, "x", ZIP_DEFLATED) as z:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in (".json", ".log"):
            z.write(p, str(p.relative_to(root)))
print("AWQ_VISION_BRANCHES: COMPLETE")
for name, values in summary.items():
    print(name, "平均动作MSE=", values["mean_action_mse"], "夹爪分歧=", values["gripper_disagreement_steps"], "/256")
print("请下载并上传：", output)
PY
