#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/trajectory-drift-20261107
while pgrep -f '^envs/oft/bin/python -u -m qvla.calibrate_multisuite_20261109_v4' >/dev/null; do sleep 30; done
grep -q 'awq PASS exit0' checks/20261109-053-calibration-status-v4.txt
grep -q 'smoothquant PASS exit0' checks/20261109-053-calibration-status-v4.txt
test "$(hostname)" = autodl-container-c49843b5ba-28bcee5c
test "$(nvidia-smi --query-gpu=uuid --format=csv,noheader)" = GPU-96b66e7f-9004-4347-7993-c799d727f664
test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0
test "$(df -B1 --output=avail /root/autodl-tmp | tail -1)" -ge 12884901888
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OMP_NUM_THREADS=4 PYTHONPATH=$PWD/code:$PWD/sources/openvla-oft:$PWD/sources/LIBERO
for condition in awq_w4a16 smoothquant_w4a8 smoothquant_w4a4; do
 method=${condition%_w*}; bits=${condition##*a}
 for suite in libero_spatial libero_object libero_goal libero_10; do
  id=20261109-053-${condition}-${suite}-task0-reset5-v1
  test ! -e runs/$id
  envs/oft/bin/python -u execute_oft_quant_drift_v2.py --oft-source sources/openvla-oft --checkpoint models/openvla_oft --suite "$suite" --task-id 0 --initial-state-index 5 --model-seed 7 --round-id "$id" --output runs/$id --profile profiles/20261109-four-suite-reset0-v4/$method.pt --official-root sources/official-quantization-pinned-v2 --method "$method" --activation-bits "$bits" > logs/$id.log 2>&1
  echo "$id exit0 PENDING_PAIRED_AUDIT" >> checks/20261109-053-quant-pilot-status-v1.txt
 done
done
