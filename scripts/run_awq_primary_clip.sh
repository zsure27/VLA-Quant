#!/usr/bin/env bash
set -Eeuo pipefail
# 只改变主视觉裁剪；语言W2无裁剪，融合视觉BF16，复用原校准profile。
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export ROOT=${ROOT:-/root/autodl-tmp/qvla-repro}
source "$HERE/scripts/activate_oft.sh"
export PYTHONPATH="$HERE:$HERE/diagnostics:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=7 WANDB_MODE=disabled
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4
export BASE=${BASE:-$ROOT/artifacts/awq-spatial-20260912-163735-1136}
export PREVIOUS=${PREVIOUS:-$ROOT/artifacts/awq-vision-branches-20260913-150727-1162}
export OUT=${OUT:-$ROOT/artifacts/awq-primary-clip-$(date +%Y%m%d-%H%M%S)-$$}
test ! -e "$OUT" || { echo "拒绝覆盖：$OUT"; exit 1; }
mkdir -p "$OUT"
echo "结果目录：$OUT"
python "$HERE/scripts/check_runtime.py" | tee "$OUT/runtime.json"
python "$HERE/diagnostics/gates.py" split --samples "$BASE/samples" --calibration-count 32 --offset 32 --count 32 | tee "$OUT/split.json"
common=(--checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial"
  --samples-dir "$BASE/samples" --official-root "$ROOT/src/official-quantization"
  --targets-file "$HERE/configs/qvla-connected-422.txt" --num-samples 32 --offset 32
  --seed 7 --attention-layers 7,15,23,31 --teacher-dir "$BASE/teacher")
run() {
  local name=$1; shift
  python "$HERE/diagnostics/probe.py" "${common[@]}" --output "$OUT/$name" "$@" 2>&1 | tee "$OUT/$name.log"
}
run repeat --mode repeat
python "$HERE/diagnostics/gates.py" control --metrics "$OUT/repeat/metrics.json" --threshold 1e-10
awq=(--mode awq --weight-bits 2 --activation-bits 16 --awq-profile "$BASE/profiles/w2.pt"
  --awq-disable-clip all --weight-scope all --awq-vision-bits 2 --awq-vision-branch primary)
run primary-original "${awq[@]}"
python - <<'PY'
import json, os, math
from pathlib import Path
a = Path(os.environ['PREVIOUS']) / 'vision-w2-primary'
b = Path(os.environ['OUT']) / 'primary-original'
old, new = [{v['sample']: v for v in json.loads((p/'metrics.json').read_text())} for p in (a,b)]
if set(old) != set(new) or len(new) != 32:
    raise RuntimeError('参照样本集合变化')
for name in new:
    x, y = [r[name]['normalized_action']['mse'] for r in (old,new)]
    if not all(map(math.isfinite, (x,y))) or abs(x-y) > 1e-8:
        raise RuntimeError('参照动作误差未复现：'+name)
    if old[name]['raw_gripper_disagreement'] != new[name]['raw_gripper_disagreement']:
        raise RuntimeError('参照夹爪指令未复现：'+name)
if json.loads((a/'manifest.json').read_text())['profiles'] != json.loads((b/'manifest.json').read_text())['profiles']:
    raise RuntimeError('profile已变化')
print('PRIMARY_REFERENCE_REPLAY: PASS')
PY
run primary-no-clip "${awq[@]}" --awq-primary-no-clip
python - <<'PY'
import json, os, statistics, math
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
root = Path(os.environ['OUT'])
summary = {}
for name in ('primary-original', 'primary-no-clip'):
    rows = json.loads((root/name/'metrics.json').read_text())
    scope = json.loads((root/name/'scope.json').read_text())
    targets = scope['weight_targets']
    if len(rows) != 32 or len({r['sample'] for r in rows}) != 32:
        raise RuntimeError('样本不完整')
    if len(targets) != 317 or sum(n.startswith('language_model.') for n in targets) != 224 or sum(n.startswith('vision_backbone.featurizer.') for n in targets) != 93:
        raise RuntimeError('量化范围不符')
    if len(scope['removed_clip_targets']) != 160 or any(not n.startswith('language_model.') for n in scope['removed_clip_targets']):
        raise RuntimeError('语言裁剪范围不符')
    removed = scope['removed_visual_clip_targets']
    if bool(removed) != (name == 'primary-no-clip') or any(not n.startswith('vision_backbone.featurizer.') for n in removed):
        raise RuntimeError('视觉裁剪范围不符')
    errors = [r['normalized_action']['mse'] for r in rows]
    if not all(map(math.isfinite, errors)):
        raise RuntimeError('动作误差非有限值')
    summary[name] = dict(mean_action_mse=statistics.mean(errors), median_action_mse=statistics.median(errors),
        max_action_mse=max(errors), gripper_disagreement_steps=round(sum(r['raw_gripper_disagreement']*8 for r in rows)),
        removed_visual_clip_count=len(removed), per_sample_action_mse={r['sample']:r['normalized_action']['mse'] for r in rows})
a,b = [summary[n]['per_sample_action_mse'] for n in ('primary-original','primary-no-clip')]
if set(a) != set(b):
    raise RuntimeError('配对样本不一致')
summary['paired'] = dict(improved=sum(b[k]<a[k] for k in a), worsened=sum(b[k]>a[k] for k in a), unchanged=sum(b[k]==a[k] for k in a))
with (root/'summary.json').open('x') as f:
    json.dump(summary,f,indent=2,allow_nan=False)
with (root/'complete.json').open('x') as f:
    json.dump(dict(status='COMPLETE',note='主视觉裁剪单因素离线诊断，不代表闭环成功率'),f)
output = Path(str(root)+'-review.zip')
with ZipFile(output,'x',ZIP_DEFLATED) as z:
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.suffix in ('.json','.log'):
            z.write(p,str(p.relative_to(root)))
print('AWQ_PRIMARY_CLIP: COMPLETE')
for name in ('primary-original','primary-no-clip'):
    print(name,summary[name]['mean_action_mse'],summary[name]['gripper_disagreement_steps'],'/256')
print('请下载并上传：',output)
PY
