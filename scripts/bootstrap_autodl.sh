#!/usr/bin/env bash
set -Eeuo pipefail

# 在新 AutoDL 实例上重建源码并安装本仓库适配代码。
ROOT=${ROOT:-/root/autodl-tmp/qvla-repro}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
INSTALL_ENV=${INSTALL_ENV:-0}
DOWNLOAD_MODEL=${DOWNLOAD_MODEL:-0}
QVLA_COMMIT=26cc4821a3be4c003d09d3c7997b38db2a347982
LIBERO_COMMIT=8f1084e3132a39270c3a13ebe37270a43ece2a01
AWQ_COMMIT=d6e797a42b9ef7778de8ee2352116e0f48a78d61
SQ_COMMIT=c61476d728e42ae0d8a35e7e78494edcac3237b5
TRANSFORMERS_COMMIT=bc339d9ad707454c0c115970db43c260067c61ab

mkdir -p "$ROOT"/{src,models,data,calib,qvla,logs,artifacts,eval}

clone_at() {
  local url=$1 destination=$2 commit=$3
  if [[ ! -d "$destination/.git" ]]; then
    git clone "$url" "$destination"
  fi
  test -z "$(git -C "$destination" status --short)" || {
    echo "源码有未提交修改，请使用新的 ROOT；不会清理或覆盖：$destination" >&2
    exit 1
  }
  git -C "$destination" fetch --tags origin
  git -C "$destination" checkout --detach "$commit"
  test -z "$(git -C "$destination" status --short)" || {
    echo "源码目录存在未记录修改，停止安装：$destination" >&2
    exit 1
  }
}

clone_at https://github.com/AutoLab-SAI-SJTU/QVLA.git "$ROOT/src/QVLA" "$QVLA_COMMIT"
clone_at https://github.com/Lifelong-Robot-Learning/LIBERO.git "$ROOT/src/LIBERO" "$LIBERO_COMMIT"
clone_at https://github.com/mit-han-lab/llm-awq.git "$ROOT/src/official-quantization/llm-awq" "$AWQ_COMMIT"
clone_at https://github.com/mit-han-lab/smoothquant.git "$ROOT/src/official-quantization/smoothquant" "$SQ_COMMIT"
clone_at https://github.com/moojink/transformers-openvla-oft.git "$ROOT/src/transformers-openvla-oft" "$TRANSFORMERS_COMMIT"

if [[ "$INSTALL_ENV" == 1 ]]; then
  # 环境创建较耗时，默认关闭。需要 conda 已可用。
  eval "$(conda shell.bash hook)"
  if ! conda env list | awk '{print $1}' | grep -qx qvla-oft; then
    conda create -n qvla-oft python=3.10.14 -y
  fi
  conda activate qvla-oft
  # 约束必须覆盖每次安装；仅第一次 requirements 固定 NumPy 不足以阻止后续升级。
  export PIP_CONSTRAINT="$HERE/constraints-oft.txt"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -r "$HERE/requirements-known.txt"
  python -m pip install -e "$ROOT/src/QVLA/openvla-oft"
  python -m pip install -e "$ROOT/src/LIBERO"
  python -m pip install -r "$ROOT/src/QVLA/openvla-oft/experiments/robot/libero/libero_requirements.txt"
  # 最后安装固定 fork，不能仅凭版本号 4.40.1 判断双向 attention 正确。
  python -m pip install --no-deps "$ROOT/src/transformers-openvla-oft"
  python -m pip check
  python "$HERE/scripts/check_runtime.py"
  # FlashAttention 对 CUDA/编译器敏感，失败时保留日志，不静默换版本。
  # 本轮使用 SDPA，不需要额外安装 FlashAttention。切换后端是单独实验。
fi

bash "$HERE/scripts/install_adapter.sh"

if [[ "$DOWNLOAD_MODEL" == 1 ]]; then
  eval "$(conda shell.bash hook)"
  conda activate qvla-oft
  MODEL_DIR="$ROOT/models/openvla-7b-oft-finetuned-libero-spatial"
  python - "$MODEL_DIR" "${HF_MODEL_REVISION:-}" <<'PY'
import sys
from huggingface_hub import snapshot_download

destination, revision = sys.argv[1:]
snapshot_download(
    repo_id="moojink/openvla-7b-oft-finetuned-libero-spatial",
    revision=revision or None,
    local_dir=destination,
)
print("模型下载完成；请将解析后的 commit SHA 写入 configs/run_manifest.json")
PY
fi

echo "源码恢复完成：$ROOT"
echo "LIBERO commit 是封装日固定值，并非已证明的历史实验 commit。"
