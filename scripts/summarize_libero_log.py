#!/usr/bin/env python3
"""解析 OpenVLA-OFT LIBERO 日志并报告二项分布置信区间。"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


EPISODES_RE = re.compile(r"Total episodes:\s*(\d+)")
SUCCESSES_RE = re.compile(r"Total successes:\s*(\d+)")
SUITE_RE = re.compile(r"Task suite:\s*(\S+)")


def wilson_interval(successes: int, episodes: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    p = successes / episodes
    denominator = 1.0 + z * z / episodes
    center = (p + z * z / (2.0 * episodes)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / episodes + z * z / (4.0 * episodes**2)) / denominator
    return center - margin, center + margin


def last_match(pattern: re.Pattern[str], text: str) -> str | None:
    matches = pattern.findall(text)
    return matches[-1] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    text = args.log.read_text(encoding="utf-8", errors="replace")
    episodes_raw = last_match(EPISODES_RE, text)
    successes_raw = last_match(SUCCESSES_RE, text)
    suite = last_match(SUITE_RE, text) or "unknown"
    if episodes_raw is None or successes_raw is None:
        raise SystemExit("Final LIBERO totals were not found; the run may be incomplete.")

    episodes = int(episodes_raw)
    successes = int(successes_raw)
    if successes > episodes:
        raise SystemExit("Invalid log: successes exceed episodes.")
    low, high = wilson_interval(successes, episodes)
    result = {
        "log": str(args.log),
        "task_suite": suite,
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "wilson_95_low": low,
        "wilson_95_high": high,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
