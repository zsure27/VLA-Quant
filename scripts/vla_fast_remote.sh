#!/usr/bin/env bash
# Run on a verified AutoDL instance. No credentials are stored or printed.
set -euo pipefail

ROOT=/root/autodl-tmp/qvla-repro
REPO=/root/VLA-Quant
EXPECTED_REMOTE=https://github.com/zsure27/VLA-Quant.git
BASE_COMMIT=852d350
PROFILE=$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w2.pt
PRIMARY=$ROOT/artifacts/awq-primary-group-20260913-213127-3342/profiles/w2-g64.pt
W4=$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w4.pt
COMMIT=$(git -C "$REPO" rev-parse --short HEAD)
BACKUP=$ROOT/backups/$(date +%Y%m%d)-$COMMIT

check() {
  test -d "$ROOT" && test -d "$REPO/.git"
  test -f "$PROFILE" && test -f "$PRIMARY"
  test "$(git -C "$REPO" remote get-url origin)" = "$EXPECTED_REMOTE"
  git -C "$REPO" merge-base --is-ancestor "$BASE_COMMIT" HEAD
  source /root/miniconda3/bin/activate /root/miniconda3/envs/qvla-oft
  python - <<'PY'
import importlib.metadata as m
for name, expected in [('robosuite', '1.4.0'), ('mujoco', '3.1.1')]:
    actual = m.version(name)
    if actual != expected:
        raise SystemExit(f'{name} version {actual}; expected {expected}')
print('Runtime and repository checks passed')
PY
}

smoke() {
  check
  cd "$REPO"
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
  export TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=2
  local run_dir="$ROOT/eval/awq-current-smoke10-$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$run_dir"
  printf 'Run directory: %s\n' "$run_dir"
  printf '%q ' python qvla/run_eval_official_quant.py --method awq --weight-bits 2 --activation-bits 16 \
    --pretrained_checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial" \
    --profile "$PROFILE" --official-root "$ROOT/src/official-quantization" \
    --task_suite_name libero_spatial --num_trials_per_task 1 --local_log_dir "$run_dir" \
    --libero_root "$ROOT/src/LIBERO" --seed 0 --env-seed 0 --seed-protocol paired \
    --awq-candidate w2-no-clip-primary-g64 --awq-primary-group64-profile "$PRIMARY" > "$run_dir/command.txt"
  printf '\n' >> "$run_dir/command.txt"
  set +e
  CUDA_VISIBLE_DEVICES=0 python qvla/run_eval_official_quant.py \
    --method awq --weight-bits 2 --activation-bits 16 \
    --pretrained_checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial" \
    --profile "$PROFILE" --official-root "$ROOT/src/official-quantization" \
    --task_suite_name libero_spatial --num_trials_per_task 1 --local_log_dir "$run_dir" \
    --libero_root "$ROOT/src/LIBERO" --seed 0 --env-seed 0 --seed-protocol paired \
    --awq-candidate w2-no-clip-primary-g64 --awq-primary-group64-profile "$PRIMARY" \
    2>&1 | tee "$run_dir/console.log"
  local code=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$code" > "$run_dir/exit-code.txt"
  (cd "$run_dir" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS.txt)
  return "$code"
}

baseline() {
  check
  source /root/miniconda3/bin/activate /root/miniconda3/envs/qvla-oft
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
  export TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=2
  export PYTHONPATH="$REPO:$ROOT/src/QVLA/openvla-oft:$ROOT/src/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
  local run_dir="$ROOT/eval/bf16-paired-smoke10-$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$run_dir"
  printf 'Run directory: %s\n' "$run_dir"
  cd "$ROOT/src/QVLA/openvla-oft"
  printf '%q ' python -m experiments.robot.libero.run_libero_eval \
    --model_family openvla --pretrained_checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial" \
    --task_suite_name libero_spatial --num_trials_per_task 1 --local_log_dir "$run_dir" \
    --center_crop True --seed 0 --env_seed 0 --seed_protocol paired --attn_implementation sdpa \
    > "$run_dir/command.txt"
  printf '\n' >> "$run_dir/command.txt"
  set +e
  CUDA_VISIBLE_DEVICES=0 python -m experiments.robot.libero.run_libero_eval \
    --model_family openvla --pretrained_checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial" \
    --task_suite_name libero_spatial --num_trials_per_task 1 --local_log_dir "$run_dir" \
    --center_crop True --seed 0 --env_seed 0 --seed_protocol paired --attn_implementation sdpa \
    2>&1 | tee "$run_dir/console.log"
  local code=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$code" > "$run_dir/exit-code.txt"
  (cd "$run_dir" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS.txt)
  return "$code"
}

w4() {
  check
  test -f "$W4"
  cd "$REPO"
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
  export TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=2
  local run_dir="$ROOT/eval/awq-w4-paired-smoke10-$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$run_dir"
  printf 'Run directory: %s\n' "$run_dir"
  printf '%q ' python qvla/run_eval_official_quant.py --method awq --weight-bits 4 --activation-bits 16 \
    --pretrained_checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial" \
    --profile "$W4" --official-root "$ROOT/src/official-quantization" \
    --task_suite_name libero_spatial --num_trials_per_task 1 --local_log_dir "$run_dir" \
    --libero_root "$ROOT/src/LIBERO" --seed 0 --env-seed 0 --seed-protocol paired \
    > "$run_dir/command.txt"
  printf '\n' >> "$run_dir/command.txt"
  set +e
  CUDA_VISIBLE_DEVICES=0 python qvla/run_eval_official_quant.py \
    --method awq --weight-bits 4 --activation-bits 16 \
    --pretrained_checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial" \
    --profile "$W4" --official-root "$ROOT/src/official-quantization" \
    --task_suite_name libero_spatial --num_trials_per_task 1 --local_log_dir "$run_dir" \
    --libero_root "$ROOT/src/LIBERO" --seed 0 --env-seed 0 --seed-protocol paired \
    2>&1 | tee "$run_dir/console.log"
  local code=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$code" > "$run_dir/exit-code.txt"
  (cd "$run_dir" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS.txt)
  return "$code"
}

backup() {
  source /root/miniconda3/bin/activate /root/miniconda3/envs/qvla-oft
  mkdir -p "$BACKUP"
  printf 'Backup directory: %s\n' "$BACKUP"
  git -C "$REPO" bundle create "$BACKUP/VLA-Quant-$COMMIT.bundle" main
  git -C "$REPO" format-patch -1 "$COMMIT" --stdout > "$BACKUP/$COMMIT.patch"
  git -C "$REPO" status --porcelain=v1 > "$BACKUP/git-status.txt"
  git -C "$REPO" diff HEAD --binary > "$BACKUP/worktree.patch"
  python -m pip freeze | grep -E '^(mujoco|numpy|robosuite)==' > "$BACKUP/runtime-lock.txt"
  printf '%s\n' "$EXPECTED_REMOTE" "$COMMIT" > "$BACKUP/pending-remote.txt"
  mkdir -p "$BACKUP/results"
  cp -a "$ROOT/eval" "$BACKUP/results/"
  diff -qr "$ROOT/eval" "$BACKUP/results/eval"
  if test -d "$REPO/rollouts"; then
    mkdir -p "$BACKUP/results/rollouts-repo"
    cp -a "$REPO/rollouts/." "$BACKUP/results/rollouts-repo/"
    diff -qr "$REPO/rollouts" "$BACKUP/results/rollouts-repo"
  fi
  if test -d "$ROOT/src/QVLA/openvla-oft/rollouts"; then
    mkdir -p "$BACKUP/results/rollouts-oft"
    cp -a "$ROOT/src/QVLA/openvla-oft/rollouts/." "$BACKUP/results/rollouts-oft/"
    diff -qr "$ROOT/src/QVLA/openvla-oft/rollouts" "$BACKUP/results/rollouts-oft"
  fi
  (cd "$BACKUP" && find results -type f -print0 | sort -z | xargs -0 -r sha256sum > RESULTS_SHA256SUMS.txt)
  (cd "$BACKUP" && sha256sum -c RESULTS_SHA256SUMS.txt)
  printf '%s\n' 'Raw evaluation files and rollout videos are on the persistent disk and in this backup; videos/model/profile weights are not uploaded to ordinary Git.' > "$BACKUP/LARGE_FILES_NOT_IN_GIT.txt"
  (cd "$BACKUP" && sha256sum "VLA-Quant-$COMMIT.bundle" "$COMMIT.patch" \
    git-status.txt worktree.patch runtime-lock.txt pending-remote.txt RESULTS_SHA256SUMS.txt LARGE_FILES_NOT_IN_GIT.txt > SHA256SUMS.txt)
  (cd "$BACKUP" && sha256sum -c SHA256SUMS.txt)
  if ! command -v gh >/dev/null 2>&1 || \
     test "$(gh api user --jq .login 2>/dev/null)" != zsure27; then
    printf '%s\n' 'GitHub identity not verified as zsure27; backup retained, push pending.'
    return 2
  fi
  if ! test "$(gh api repos/zsure27/VLA-Quant --jq .permissions.push 2>/dev/null)" = true; then
    printf '%s\n' 'GitHub write permission not verified; push pending.'
    return 2
  fi
  if ! GIT_TERMINAL_PROMPT=0 git -C "$REPO" -c credential.helper= \
      -c 'credential.helper=!gh auth git-credential' push origin main; then
    printf '%s\n' 'GitHub push pending: retain the persistent backup and all experiment data.'
    return 2
  fi
  git -C "$REPO" fetch origin main
  test "$(git -C "$REPO" rev-parse origin/main)" = "$(git -C "$REPO" rev-parse main)"
  printf '%s\n' 'GitHub main matches local main.'
}

case "${1:-}" in
  check) check ;;
  smoke) smoke ;;
  baseline) baseline ;;
  w4) w4 ;;
  backup) backup ;;
  *) printf 'Usage: %s check|smoke|baseline|w4|backup\n' "$0" >&2; exit 2 ;;
esac
