"""One-shot 046 dispatcher: wait for running R0 gate, register, then launch.

No quota API access is claimed. The active 15-minute client heartbeat owns live
quota checks and STOP_NEW_GPU. Never starts a completed or existing new plan.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", type=Path, required=True)
    a = p.parse_args()
    c = a.control.resolve()
    if socket.gethostname() != "autodl-container-90824aab47-6dc00ae3":
        raise ValueError("Wrong 046 hostname")
    receipt_path = c / "B3-dispatch-v1.json"
    with (c / "B3-dispatch-v1.once").open("x") as stream:
        stream.write(str(os.getpid()) + "\n")
    source = c / "source-v2"
    session = Path("/root/autodl-tmp/qvla-repro/backups/experiments/p2-shared-peft/20261015-046-quant-finetuning-v1")
    gate = session / "actual-observation-repeatability-v2.json"
    result = {"state": "WAITING_FOR_RUNNING_R0_GATE", "dispatcher_pid": os.getpid(),
              "source_root": str(source), "gpu_experiment_started": False}
    def save():
        result["time_utc"] = datetime.now(timezone.utc).isoformat()
        temp = receipt_path.with_suffix(".tmp")
        temp.write_text(json.dumps(result, indent=2) + "\n")
        temp.replace(receipt_path)
    try:
        save()
        deadline = time.monotonic() + 1200
        while not gate.exists():
            if (c / "STOP_NEW_GPU").exists():
                raise RuntimeError("Client requested no new GPU work")
            if time.monotonic() > deadline:
                raise TimeoutError("Running R0 gate did not finish in bounded wait")
            status = json.loads((c / "control/20261015-046-quant-finetuning-v1-R0-EGL-v2/status.json").read_text())
            if status["state"] == "FAILED":
                raise RuntimeError("R0 runner failed; no GPU admission")
            time.sleep(10)
        if json.loads(gate.read_text())["gate"] != "PASS_CPU_PROTOCOL_OBSERVATION_BRIDGE":
            raise ValueError("Actual observation repeatability gate failed")
        if (c / "STOP_NEW_GPU").exists():
            raise RuntimeError("Client requested no new GPU work")
        if shutil.disk_usage(session).free < 12 * 1024**3:
            raise ValueError("Need 12GiB before GPU start; retain 6GiB during run")
        active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
        if active:
            raise ValueError("GPU already has a compute process; do not duplicate")
        manifest = json.loads((c / "source-v2-SHA256.json").read_text())
        if any(hashlib.sha256((source / name).read_bytes()).hexdigest() != digest for name, digest in manifest.items()):
            raise ValueError("Deployed source changed after CPU admission")
        py = "/root/miniconda3/envs/qvla-oft/bin/python"
        materials = c / "materials-b3-initialization-v1.json"
        m = json.loads(materials.read_text())
        env = os.environ.copy()
        env["PYTHONPATH"] = ":".join((str(source), str(source / "diagnostics"), m["oft_root"], m["libero_root"]))
        command = [py, str(source / "scripts/prepare_b3_initialization_plans_v1.py"),
                   "--materials", str(materials), "--output", str(c / "B3-plans-v1"), "--session", str(session),
                   "--resume-gate", str(session / "resume-content-gate.json"), "--observation-gate", str(gate),
                   "--cpu-gate", str(c / "cpu-contracts-v3.json"), "--expected-hostname", socket.gethostname(), "--python", py]
        with (c / "B3-registration-v1.log").open("xb") as log:
            subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        plan = c / "B3-plans-v1/first50-plan.json"
        value = json.loads(plan.read_text())
        result.update({"state": "GPU_PLAN_LAUNCHING", "plan": str(plan),
                       "plan_file_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
                       "source_tar_sha256": "7e46f0403260a0c0297496c57050af9a11f75a7d9cdadb82f686a376c7594c1d",
                       "registered_source_commit": "fc3e451d4c42de0b44fb7c36562531079e651eed",
                       "status": str(c / "B3-plans-v1/control" / value["plan_id"] / "status.json")})
        save()
        with (c / "B3-supervisor-v1.log").open("xb") as log:
            child = subprocess.Popen([py, str(source / "scripts/vla_stage_supervisor.py"), str(plan)],
                                     cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        result.update({"state": "GPU_PLAN_SUPERVISOR_STARTED", "supervisor_pid": child.pid,
                       "gpu_experiment_started": "supervisor started; inspect actual stage/GPU process for execution"})
        save()
    except Exception as exc:
        result.update({"state": "FAILED_NO_AUTOMATIC_RETRY", "error_type": type(exc).__name__, "error": str(exc)})
        save()
        raise


if __name__ == "__main__":
    main()
