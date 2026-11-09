#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/trajectory-drift-20261107
test "$(hostname)" = autodl-container-c49843b5ba-28bcee5c
test "$(nvidia-smi --query-gpu=uuid --format=csv,noheader)" = GPU-96b66e7f-9004-4347-7993-c799d727f664
test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0
test "$(df -B1 --output=avail /root/autodl-tmp | tail -1)" -ge 8589934592
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OMP_NUM_THREADS=4 PYTHONPATH=$PWD/code:$PWD/sources/openvla-oft:$PWD/sources/LIBERO
for condition in bf16 awq_w4a16 smoothquant_w4a8 smoothquant_w4a4; do
 for suite in libero_spatial libero_object libero_goal libero_10; do
  for task in {1..9}; do
   if test -e control/STOP_NEW_GPU_20261109; then echo STOP_NEW_GPU >> checks/20261109-053-task1-9-status-v1.txt; exit 90; fi
   test "$(df -B1 --output=avail /root/autodl-tmp | tail -1)" -ge 6442450944
   id=20261109-053-${condition}-${suite}-task${task}-reset5-v1
   test ! -e runs/$id
   args=(--oft-source sources/openvla-oft --checkpoint models/openvla_oft --suite "$suite" --task-id "$task" --initial-state-index 5 --model-seed 7 --round-id "$id" --output runs/$id)
   if test "$condition" = bf16; then
    envs/oft/bin/python -u -m qvla.execute_oft_drift_v3 "${args[@]}" > logs/$id.log 2>&1
   else
    method=${condition%_w*}; bits=${condition##*a}
    envs/oft/bin/python -u execute_oft_quant_drift_v2.py "${args[@]}" --profile profiles/20261109-four-suite-reset0-v4/$method.pt --official-root sources/official-quantization-pinned-v2 --method "$method" --activation-bits "$bits" > logs/$id.log 2>&1
   fi
   echo "$id exit0 PENDING_PAIRED_AUDIT" >> checks/20261109-053-task1-9-status-v1.txt
  done
 done
done
