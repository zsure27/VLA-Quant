#!/usr/bin/env bash
set -Eeuo pipefail

# 原服务器恢复后执行：仅生成清单，不复制、不删除大文件。
ROOT=${ROOT:-/root/autodl-tmp/qvla-repro}
OUTPUT=${OUTPUT:-$ROOT/artifacts/server-data-inventory.tsv}
mkdir -p "$(dirname "$OUTPUT")"
printf 'sha256\tbytes\tpath\n' > "$OUTPUT"
for relative in \
  models/openvla-7b-oft-finetuned-libero-spatial \
  calib/action-space-balanced/libero-512 \
  qvla/spatial/official-w4-vl \
  logs/official-quant-validation \
  artifacts/official-quant-validation; do
  target=$ROOT/$relative
  if [[ -d "$target" ]]; then
    find "$target" -type f -print0 | sort -z | while IFS= read -r -d '' file; do
      hash=$(sha256sum "$file" | awk '{print $1}')
      bytes=$(stat -c '%s' "$file")
      printf '%s\t%s\t%s\n' "$hash" "$bytes" "${file#"$ROOT/"}" >> "$OUTPUT"
    done
  fi
done
echo "清单已保存：$OUTPUT"
