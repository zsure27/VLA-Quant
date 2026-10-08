"""Execute one argv stage against pinned code/materials and the current hostname."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qvla.extended_peft import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lock", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    lock = json.loads(a.lock.read_text())
    if socket.gethostname() != lock["expected_hostname"]:
        raise ValueError("Instance hostname differs from this run; never reuse another instance plan")
    for name, expected in lock["files"].items():
        if sha(name) != expected: raise ValueError(f"Locked input/code changed: {name}")
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command: raise ValueError("Empty command")
    start_file = a.lock.with_suffix(".started.json")
    try:
        with start_file.open("x") as stream:
            json.dump({"started_unix": time.time(), "hostname": socket.gethostname()}, stream)
    except FileExistsError:
        pass
    start = json.loads(start_file.read_text())
    if start["hostname"] != socket.gethostname(): raise ValueError("Plan budget belongs to another host")
    remaining = lock["max_plan_seconds"] - (time.time() - start["started_unix"])
    if remaining <= 0: raise TimeoutError("Preregistered total plan time budget exhausted")
    if a.output.exists(): raise ValueError("Stage output exists; audit recovery, do not repeat/overwrite")
    a.output.mkdir(parents=True)
    (a.output / "invocation.json").write_text(json.dumps({"command": command, "lock_sha256": sha(a.lock),
        "hostname": socket.gethostname(), "started_unix": time.time()}, indent=2) + "\n")
    with (a.output / "console.log").open("wb") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=os.name != "nt")
        try:
            code = child.wait(timeout=min(lock["max_stage_seconds"], remaining))
        except BaseException:
            if os.name != "nt": os.killpg(child.pid, signal.SIGTERM)
            else: child.terminate()
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if os.name != "nt": os.killpg(child.pid, signal.SIGKILL)
                else: child.kill()
                child.wait()
            code = 124
    (a.output / "exit-code.txt").write_text(str(code) + "\n")
    raise SystemExit(code)


if __name__ == "__main__": main()
