#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/trajectory-drift-20261107
test "$(hostname)" = autodl-container-c49843b5ba-28bcee5c
test "$(nvidia-smi --query-gpu=uuid --format=csv,noheader)" = GPU-96b66e7f-9004-4347-7993-c799d727f664
test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0
test "$(df -B1 --output=avail /root/autodl-tmp | tail -1)" -ge 12884901888
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OMP_NUM_THREADS=4 PYTHONPATH=$PWD/code:$PWD/sources/openvla-oft:$PWD/sources/LIBERO
mkdir -p profiles/20261109-four-suite-reset0-v2
for method in awq smoothquant; do
 envs/oft/bin/python -u calibrate_guarded_20261109_v2.py --method "$method" --checkpoint models/openvla_oft --samples-dir calibration/20261109-four-suite-reset0-v1 --targets-file checks/connected422-20261109.txt --official-root sources/official-quantization-pinned-v2 --output profiles/20261109-four-suite-reset0-v2/$method.pt --summary-json profiles/20261109-four-suite-reset0-v2/$method.json --num-samples 4 --bits 4 --max-rows 64 --max-output-rows 256 --clip-tokens 64 > logs/20261109-calibration-$method-v2.log 2>&1
 echo "$method PASS exit0" >> checks/20261109-053-calibration-status-v2.txt
done
