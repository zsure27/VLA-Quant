#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/trajectory-drift-20261107
export PYTHONPATH=$PWD/sources/openpi/src:$PWD/sources/openpi/packages/openpi-client/src OMP_NUM_THREADS=4
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu envs/pi05/bin/python -u audit_pi05_conversion_mapping_20261109_v2.py > logs/20261109-pi05-mapping-audit-v2.log 2>&1
test "$(nvidia-smi --query-gpu=uuid --format=csv,noheader)" = GPU-3e150638-1b75-05b2-0246-a4bd02bb7723
test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cpu envs/pi05/bin/python -u pi05_same_input_bridge_v4.py --backend torch --query runs/20261108-216-pi05-spatial-bf16-task0-reset5-v2/query-00000.npz --openpi-source sources/openpi --checkpoint models/pi05_libero_pytorch_bridge_v2 --native-checkpoint models/pi05_libero --jax-reference checks/pi05-same-device-jax-v3 --output checks/pi05-same-input-torch-v4 > logs/20261109-pi05-torch-bridge-v4.log 2>&1
