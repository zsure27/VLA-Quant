#!/usr/bin/env bash
set -Eeuo pipefail
# 修复依赖漂移与 CRLF/LF 哈希错误；不重克隆、不删除模型/数据、不跳过校验。
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=${ROOT:-/root/autodl-tmp/qvla-repro}
source "$HERE/scripts/activate_oft.sh"
python - <<'PY'
import sys
print("当前解释器：", sys.executable)
if sys.version_info[:2] != (3, 10):
    raise SystemExit("要求 Python 3.10；请检查 qvla-oft 环境，不能修改镜像基础 Python 3.12")
PY
REPAIR_LOG=$ROOT/artifacts/install-repair-$(date +%Y%m%d-%H%M%S)
mkdir -p "$REPAIR_LOG"
python -m pip freeze > "$REPAIR_LOG/pip-freeze-before.txt"
export PIP_CONSTRAINT="$HERE/constraints-oft.txt"
# OpenCV 5 在 Python>=3.9 要求 NumPy>=2；两者必须联合调整。
# 当前日志中的 Numba 0.67 声明兼容 NumPy 1.26，不因此额外降级 Numba/llvmlite。
python -m pip install "numpy==1.26.4" "opencv-python==4.10.0.84" 2>&1 | tee "$REPAIR_LOG/pip-repair.log"
python -m pip check 2>&1 | tee "$REPAIR_LOG/pip-check.log"
python - <<'PY'
import numpy, cv2, numba, torch, tensorflow as tf
print("numpy", numpy.__version__, "opencv", cv2.__version__, "numba", numba.__version__)
print("torch", torch.__version__, "torch CUDA", torch.version.cuda, "tensorflow", tf.__version__)
assert numpy.__version__ == "1.26.4"
assert torch.__version__.split('+')[0] == "2.2.0"
assert tf.__version__ == "2.15.0"
print("DEPENDENCY_IMPORT: PASS")
PY
python "$HERE/scripts/check_runtime.py" | tee "$REPAIR_LOG/runtime.log"
bash "$HERE/scripts/install_adapter.sh" 2>&1 | tee "$REPAIR_LOG/install-adapter.log"
python -m pip freeze > "$REPAIR_LOG/pip-freeze-after.txt"
echo "INSTALL_REPAIR: PASS。日志：$REPAIR_LOG"
echo "本脚本未下载模型/数据，也未执行 GPU 量化与 rollout。"
