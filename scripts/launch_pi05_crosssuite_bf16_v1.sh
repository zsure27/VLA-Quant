#!/usr/bin/env bash
# Run three preregistered native pi0.5 BF16 suite admissions sequentially.
set -euo pipefail

ROOT=/root/autodl-tmp/trajectory-drift-20261107
cd "$ROOT"
test "$(hostname)" = autodl-container-75c54eac1e-02156c9d
test "$(git -C code rev-parse HEAD)" = 2dffac3cccc0de632e571d156b163053413c30f4
test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0
test "$(df -B1 --output=avail /root/autodl-tmp | tail -1)" -ge 8589934592
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=code:sources/openpi/src:sources/openpi/packages/openpi-client/src:sources/LIBERO

for suite in libero_object libero_goal libero_10; do
    id="20261108-216-pi05-${suite}-bf16-task0-reset5-v1"
    plan="code/Ex_Reports/_tasks/multisuite-trajectory-deviation-v1/plans/${id}.json"
    output="runs/${id}"
    test ! -e "$output"
    set +e
    envs/pi05/bin/python -u -m qvla.execute_pi05_bf16_episode_v1 \
        --checkpoint models/pi05_libero \
        --openpi-source sources/openpi \
        --libero-source sources/LIBERO \
        --libero-config libero-config-v1 \
        --plan "$plan" --output "$output" \
        --round-id "$id" --suite "$suite" --task-id 0 --reset-index 5 --model-seed 7 \
        > "logs/${id}.log" 2>&1
    rc=$?
    set -e
    printf 'exit:%d\n' "$rc" >> "logs/${id}.log"
    if [ "$rc" -ne 0 ]; then
        printf '{"suite":"%s","round_id":"%s","status":"RUNNER_FAILED","exit_code":%d}\n' "$suite" "$id" "$rc" \
            >> checks/216-pi05-crosssuite-stage-status-v1.jsonl
        exit "$rc"
    fi
    set +e
    envs/pi05/bin/python -m qvla.audit_pi05_native_trace_v1 \
        --root "$output" --plan "$plan" --output "checks/${id}-audit.json" \
        > "logs/${id}-audit.log" 2>&1
    audit_rc=$?
    set -e
    if [ "$audit_rc" -ne 0 ]; then
        printf '{"suite":"%s","round_id":"%s","status":"AUDIT_FAILED","exit_code":%d}\n' "$suite" "$id" "$audit_rc" \
            >> checks/216-pi05-crosssuite-stage-status-v1.jsonl
        exit "$audit_rc"
    fi
    printf '{"suite":"%s","round_id":"%s","status":"PASS_PI05_NATIVE_BF16_TRACE_AUDIT"}\n' "$suite" "$id" \
        >> checks/216-pi05-crosssuite-stage-status-v1.jsonl
done
