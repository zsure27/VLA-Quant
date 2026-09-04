#!/usr/bin/env python3
"""不增加第三方依赖，测量命令墙钟时间并抽样 NVIDIA 显存。"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path


def read_gpu_memory() -> list[int]:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [int(line.strip()) for line in output.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-seconds", type=float, default=0.2)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("Provide a command after --")
    if args.sample_seconds <= 0:
        raise SystemExit("--sample-seconds must be positive")

    stop = threading.Event()
    samples: list[dict[str, object]] = []

    def sample() -> None:
        while not stop.is_set():
            samples.append({"time": time.time(), "memory_mib": read_gpu_memory()})
            stop.wait(args.sample_seconds)

    worker = threading.Thread(target=sample, daemon=True)
    started = time.time()
    worker.start()
    process = subprocess.run(command, check=False)
    finished = time.time()
    stop.set()
    worker.join(timeout=max(1.0, args.sample_seconds * 2.0))

    gpu_count = max((len(sample["memory_mib"]) for sample in samples), default=0)
    peaks = [0] * gpu_count
    for sample in samples:
        for index, value in enumerate(sample["memory_mib"]):
            peaks[index] = max(peaks[index], value)

    result = {
        "command": command,
        "returncode": process.returncode,
        "started_unix": started,
        "finished_unix": finished,
        "wall_seconds": finished - started,
        "gpu_peak_memory_mib": peaks,
        "sample_count": len(samples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
