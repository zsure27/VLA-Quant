#!/usr/bin/env python3
"""校验 QVLA 发布版代理脚本读取的图像/文本 JSONL。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--require", type=int, default=512)
    args = parser.parse_args()

    valid = 0
    invalid: list[str] = []
    with args.jsonl.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                invalid.append(f"line {line_number}: invalid JSON ({exc.msg})")
                continue
            if not isinstance(item, dict):
                invalid.append(f"line {line_number}: expected object")
                continue
            if not isinstance(item.get("text"), str) or not item["text"].strip():
                invalid.append(f"line {line_number}: missing non-empty text")
                continue
            image = item.get("image")
            if not isinstance(image, str) or not Path(image).is_file():
                invalid.append(f"line {line_number}: image does not exist: {image!r}")
                continue
            valid += 1

    print(f"valid samples: {valid}")
    print(f"invalid samples: {len(invalid)}")
    for message in invalid[:20]:
        print(message)
    if len(invalid) > 20:
        print(f"... {len(invalid) - 20} more invalid lines")
    if invalid or valid < args.require:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
