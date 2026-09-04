#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-artifacts/environment}"
mkdir -p "${OUT_DIR}"

date --iso-8601=seconds > "${OUT_DIR}/timestamp.txt"
uname -a > "${OUT_DIR}/uname.txt"
nvidia-smi -q > "${OUT_DIR}/nvidia-smi-q.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,power.limit --format=csv,noheader \
  > "${OUT_DIR}/gpu.csv"
python --version > "${OUT_DIR}/python.txt" 2>&1
python -m pip freeze > "${OUT_DIR}/pip-freeze.txt"

python - <<'PY' > "${OUT_DIR}/torch.txt"
import json
import torch

result = {
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cudnn": torch.backends.cudnn.version(),
}
if torch.cuda.is_available():
    result.update(
        {
            "device_name": torch.cuda.get_device_name(0),
            "capability": torch.cuda.get_device_capability(0),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
    )
print(json.dumps(result, indent=2))
PY

for repo in QVLA LIBERO; do
  if [[ -d "src/${repo}/.git" ]]; then
    git -C "src/${repo}" rev-parse HEAD > "${OUT_DIR}/${repo}-commit.txt"
    git -C "src/${repo}" status --short > "${OUT_DIR}/${repo}-status.txt"
  fi
done

echo "Environment fingerprint saved to ${OUT_DIR}"
