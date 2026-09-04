#!/usr/bin/env bash
set -Eeuo pipefail

# 将维护代码安装到固定版本 QVLA checkout；默认应用历史服务器的三份模型覆盖文件。
ROOT=${ROOT:-/root/autodl-tmp/qvla-repro}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OFT=${OFT:-$ROOT/src/QVLA/openvla-oft}
APPLY_MODEL_OVERLAY=${APPLY_MODEL_OVERLAY:-1}
test -d "$OFT/qvla" || { echo "找不到 QVLA OpenVLA-OFT：$OFT" >&2; exit 1; }

mkdir -p "$OFT/qvla" "$ROOT/scripts" "$ROOT/artifacts/awq-vl-w2-target-groups"
cp "$HERE"/qvla/*.py "$OFT/qvla/"
cp "$HERE/configs/qvla-connected-422.txt" "$ROOT/artifacts/qvla-connected-422.txt"
cp "$HERE/scripts/run-official-w4-smoke20.sh" "$ROOT/scripts/"
cp "$HERE/scripts/run-official-quant-validation.sh" "$ROOT/scripts/"
cp "$HERE/scripts/capture_env.sh" "$ROOT/scripts/"

if [[ "$APPLY_MODEL_OVERLAY" == 1 ]]; then
  check_and_copy() {
    local relative=$1 expected=$2 source=$HERE/overlays/openvla-oft/$relative destination=$OFT/$relative
    test -f "$destination" || { echo "缺少上游文件：$destination" >&2; exit 1; }
    local actual
    actual=$(sha256sum "$destination" | awk '{print toupper($1)}')
    if [[ "$actual" != "$expected" ]]; then
      echo "上游文件 SHA256 与封装基线不符，拒绝覆盖：$relative" >&2
      echo "expected=$expected actual=$actual" >&2
      exit 1
    fi
    cp "$source" "$destination"
  }
  check_and_copy prismatic/extern/hf/modeling_prismatic.py F6ABA7898AED7A57405C1D68343086243C34506359F35DA7625C9018EF943938
  check_and_copy experiments/robot/openvla_utils.py 6C25918F5EA2318C99E147325C5E601206271CC2662C65467C0F1F2F8DA0E5F3
  check_and_copy experiments/robot/libero/run_libero_eval.py 61221D0AC03F8F5F8C3DD9264DF529FDCB6DAAACE4FF8CE3808AF29F4D374F1F
fi

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
