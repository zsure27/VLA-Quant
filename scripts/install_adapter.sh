#!/usr/bin/env bash
set -Eeuo pipefail

# 将维护代码安装到固定版本 QVLA checkout；默认应用历史服务器的三份模型覆盖文件。
ROOT=${ROOT:-/root/autodl-tmp/qvla-repro}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OFT=${OFT:-$ROOT/src/QVLA/openvla-oft}
APPLY_MODEL_OVERLAY=${APPLY_MODEL_OVERLAY:-1}
test -d "$OFT/qvla" || { echo "找不到 QVLA OpenVLA-OFT：$OFT" >&2; exit 1; }

# 在复制任何工具前预检，不能把半安装环境当作已完成。
if [[ "$APPLY_MODEL_OVERLAY" == 1 ]]; then
  python "$HERE/scripts/install_overlays.py" --source "$HERE/overlays/openvla-oft" \
    --destination "$OFT" --backup-root "$ROOT/artifacts" --check-only
fi

mkdir -p "$OFT/qvla" "$ROOT/scripts" "$ROOT/artifacts/awq-vl-w2-target-groups"
cp "$HERE/configs/qvla-connected-422.txt" "$ROOT/artifacts/qvla-connected-422.txt"
cp "$HERE/scripts/run-official-w4-smoke20.sh" "$ROOT/scripts/"
cp "$HERE/scripts/run-official-quant-validation.sh" "$ROOT/scripts/"
cp "$HERE/scripts/capture_env.sh" "$ROOT/scripts/"

if [[ "$APPLY_MODEL_OVERLAY" == 1 ]]; then
  python "$HERE/scripts/install_overlays.py" --source "$HERE/overlays/openvla-oft" \
    --destination "$OFT" --backup-root "$ROOT/artifacts"
fi

# 保存原 qvla 工具后再安装维护版本，避免丢失服务器上的历史代码。
BACKUP=$ROOT/artifacts/qvla-before-install-$(date +%Y%m%d-%H%M%S)
test ! -e "$BACKUP" || { echo "备份路径已存在：$BACKUP"; exit 1; }
mkdir -p "$BACKUP"
cp -a "$OFT/qvla/." "$BACKUP/"
cp "$HERE"/qvla/*.py "$OFT/qvla/"

# 从完整目标名单生成与历史脚本相同的 9 个 shard，避免依赖服务器遗留文件。
python - "$ROOT/artifacts/qvla-connected-422.txt" "$ROOT/artifacts/awq-vl-w2-target-groups" <<'PY'
import re
import sys
from pathlib import Path

source, output = map(Path, sys.argv[1:])
names = [line.strip() for line in source.read_text().splitlines() if line.strip()]
assert len(names) == len(set(names)) == 422
groups = {"00-vision": [name for name in names if name.startswith("vision_backbone.")]}
for start in range(0, 32, 4):
    stop = start + 3
    pattern = re.compile(r"language_model\.model\.layers\.(\d+)\.")
    groups[f"language-{start:02d}-{stop:02d}"] = [
        name for name in names
        if (match := pattern.match(name)) and start <= int(match.group(1)) <= stop
    ]
output.mkdir(parents=True, exist_ok=True)
for name, values in groups.items():
    (output / f"{name}.txt").write_text("\n".join(values) + "\n")
assert sum(map(len, groups.values())) == 422
print({name: len(values) for name, values in groups.items()})
PY

echo "适配代码安装完成。legacy/ 中的历史 W2 脚本未安装。"
