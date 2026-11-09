#!/usr/bin/env bash
# Isolated official JAX -> PyTorch conversion for numerical admission only.
set -euo pipefail

ROOT=/root/autodl-tmp/trajectory-drift-20261107
cd "$ROOT"
test "$(hostname)" = autodl-container-75c54eac1e-02156c9d
test "$(git -C sources/openpi rev-parse HEAD)" = 215abfb217dbac7d5f1273282331b9b1866c0479
test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0
test "$(df -B1 --output=avail /root/autodl-tmp | tail -1)" -ge 17179869184
test "$(free -b | awk '/^Mem:/ {print $7}')" -ge 34359738368

SOURCE="$ROOT/models/pi05_libero"
OUTPUT="$ROOT/models/pi05_libero_pytorch_bridge_v1"
test -f "$SOURCE/DOWNLOAD_VERIFICATION.json"
test ! -e "$OUTPUT"

# The upstream converter restores the JAX checkpoint to CPU NumPy first. Its
# converted BF16 output is never admitted until the independent same-input
# numerical bridge succeeds. Keep any partial output for failure forensics.
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=4
export PYTHONPATH="$ROOT/sources/openpi/src:$ROOT/sources/openpi/packages/openpi-client/src"
envs/pi05/bin/python sources/openpi/examples/convert_jax_model_to_pytorch.py \
    --checkpoint-dir "$SOURCE" --config-name pi05_libero \
    --output-path "$OUTPUT" --precision bfloat16
test -s "$OUTPUT/model.safetensors"
sha256sum "$OUTPUT/model.safetensors" > checks/pi05-pytorch-bridge-v1.sha256
du -B1 -s "$OUTPUT" > checks/pi05-pytorch-bridge-v1.bytes
