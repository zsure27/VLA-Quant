#!/usr/bin/env bash
set -Eeuo pipefail
# 独立新输出目录；不覆盖已有 profile、样本或教师缓存。
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export ROOT=${ROOT:-/root/autodl-tmp/qvla-repro}
source "$HERE/scripts/activate_oft.sh"
export PYTHONPATH="$HERE:$HERE/diagnostics:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=7 WANDB_MODE=disabled
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4
export BASE=${BASE:-$ROOT/artifacts/awq-spatial-20260912-163735-1136}
export OUT=${OUT:-$ROOT/artifacts/awq-block-audit-$(date +%Y%m%d-%H%M%S)-$$}
test ! -e "$OUT" || { echo "拒绝覆盖：$OUT"; exit 1; }
python "$HERE/diagnostics/awq_block_audit.py" \
  --base "$BASE" --checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial" \
  --official-root "$ROOT/src/official-quantization" --output "$OUT" \
  --layers 10,11,12,25,26 --samples 35,48,57 \
  2>&1 | tee "$OUT.console.log"
python - <<'PY'
import os
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
root = Path(os.environ["OUT"])
files = [root / name for name in ("manifest.json", "controls.json", "metrics.json", "complete.json")]
files.append(Path(str(root) + ".console.log"))
for path in files:
    if not path.is_file():
        raise FileNotFoundError(path)
output = Path(str(root) + "-review.zip")
with ZipFile(output, "x", ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, path.name)
print("请下载并上传：", output)
PY
