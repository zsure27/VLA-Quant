#!/usr/bin/env bash
set -Eeuo pipefail
echo "此历史自动 500 回合入口已停用，避免复用旧 profile/结果；请使用 scripts/run_audit.sh 与 v3 评估入口。" >&2
exit 2
