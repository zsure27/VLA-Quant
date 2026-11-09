#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/trajectory-drift-20261107
while kill -0 3847 2>/dev/null; do sleep 30; done
test "$(wc -l < checks/20261109-053-quant-pilot-status-v1.txt)" -eq 12
test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OMP_NUM_THREADS=4 PYTHONPATH=$PWD/code:$PWD/sources/openvla-oft:$PWD/sources/LIBERO
for suite in libero_spatial libero_object libero_goal libero_10; do
 id=20261109-053-bf16-repeat-${suite}-task0-reset5-v1
 envs/oft/bin/python -u -m qvla.execute_oft_drift_v3 --oft-source sources/openvla-oft --checkpoint models/openvla_oft --suite "$suite" --task-id 0 --initial-state-index 5 --model-seed 7 --round-id "$id" --output runs/$id > logs/$id.log 2>&1
 echo "$id exit0" >> checks/20261109-053-bf16-repeat-status-v1.txt
done
