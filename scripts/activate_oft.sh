#!/usr/bin/env bash
# 兼容原虚拟环境与新 conda 环境；调用者用 source 加载。
ROOT=${ROOT:-/root/autodl-tmp/qvla-repro}
if [[ -f "$ROOT/envs/qvla-oft/bin/activate" ]]; then
  source "$ROOT/envs/qvla-oft/bin/activate"
else
  eval "$(conda shell.bash hook)"
  conda activate qvla-oft
fi
if [[ -f "$ROOT/env.sh" ]]; then source "$ROOT/env.sh"; fi
export PYTHONHASHSEED=${SEED:-7}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT/src/QVLA/openvla-oft:$ROOT/src/LIBERO:${PYTHONPATH:-}"
