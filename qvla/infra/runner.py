"""Single-server, single-model executor with low frequency status and guaranteed closure attempts."""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
from .common import append_jsonl, canonical, now, read_json, write_json
from .config import require
from .lifecycle import native_shutdown, quota_decision
from .plan import expand, verify_plan
from .report import record_round


def stop_child(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)


def run_command(argv, cwd, log_path, timeout):
    with Path(log_path).open("ab") as stream:
        process = subprocess.Popen(argv, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            stop_child(process)
            raise
    require(code == 0, f"Closure command failed with exit {code}; see private run log")


def observe(run_dir):
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    require(result.returncode == 0, "GPU observation failed")
    return {"hostname": socket.gethostname(), "gpu": result.stdout.strip(),
            "disk_free_bytes": shutil.disk_usage(run_dir).free}


def execute(plan, authorization_path, round_id, quota_path):
    config = plan["config"]
    auth = read_json(authorization_path)
    # A previous round's authorization cannot turn a clone/old RUNNING status into live permission.
    require(os.name == "posix", "Execution is only supported on the user-opened Linux server")
    require(auth.get("authorized") is True and auth.get("shutdown_authorized") is True and
            auth.get("server_id") == config["control"]["server_id"] and
            auth.get("hostname") == socket.gethostname() and
            auth.get("config_sha256") == canonical(config) and auth.get("round_id") == round_id,
            "Current round authorization/hostname/config mismatch; no process started")
    expires = datetime.fromisoformat(auth["expires_at"].replace("Z", "+00:00"))
    require(datetime.now(timezone.utc) < expires, "Round authorization expired")
    run_dir, repo = Path(plan["run_dir"]), Path(plan["repo"])
    persistent = Path(config["control"].get("persistent_root", "/root/autodl-tmp/qvla-repro")).resolve()
    require(persistent != Path(persistent.anchor) and persistent in run_dir.resolve().parents,
            "Run outputs must be inside the configured persistent server root")
    run_dir.mkdir(parents=True, exist_ok=True)
    # Refuse a duplicate launch, including after interruption. Continue with a new round id/directory.
    fd = os.open(run_dir / "execution.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    events = run_dir / "events.jsonl"
    control = config["control"]
    context = {"repo": str(repo), "run_dir": str(run_dir), "python": __import__("sys").executable,
               "task_id": config["task"]["id"], "round_id": round_id}
    child = None
    state = {"round_id": round_id, "config_sha256": plan["config_sha256"],
             "plan_sha256": plan["plan_sha256"], "started_at": now(), "state": "PREFLIGHT", "stages": []}
    requested = []
    old_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, lambda signum, _frame: requested.append(signum))

    def publish(event, **values):
        state.update(values, updated_at=now())
        write_json(run_dir / "status.json", state)
        append_jsonl(events, {"time_utc": now(), "event": event, **values})

    def quota():
        return quota_decision(read_json(quota_path), control)

    try:
        write_json(run_dir / "plan.json", plan)
        for source in plan["source_sha256"]:
            path = Path(source)
            destination = run_dir / "source-snapshot" / path.relative_to(repo)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        verify_plan(plan)
        actual = observe(run_dir)
        require(actual["disk_free_bytes"] >= control.get("start_disk_gib", 12) * 1024**3, "Insufficient start disk")
        gpu = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True, timeout=30)
        require(not gpu.strip(), "GPU already occupied")
        require(quota() == "CONTINUE", "No quota for a new GPU expansion")
        publish("admitted", actual=actual, state="RUNNING")
        for stage in plan["stages"]:
            if requested or quota() != "CONTINUE":
                publish("stop-new-stages", state="CLOSING", reason="signal_or_quota_no_expansion")
                break
            Path(run_dir / stage["id"]).mkdir(exist_ok=True)
            started = time.monotonic()
            entry = {"id": stage["id"], "argv": stage["argv"], "started_at": now()}
            state["stages"].append(entry)
            with (run_dir / (stage["id"] + ".stdout.log")).open("xb") as stream:
                env = dict(os.environ, PYTHONHASHSEED=str(config["evaluation"].get("seed", 7)),
                           WANDB_MODE="disabled", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
                           CUBLAS_WORKSPACE_CONFIG=":4096:8", TOKENIZERS_PARALLELISM="false")
                extra_paths = [str(repo), config["model"].get("oft_root", ""), config["model"].get("libero_root", "")]
                env["PYTHONPATH"] = os.pathsep.join([p for p in extra_paths if p] + [env.get("PYTHONPATH", "")])
                entry["env"] = {k: env[k] for k in ("PYTHONHASHSEED", "WANDB_MODE", "MUJOCO_GL",
                                 "PYOPENGL_PLATFORM", "CUBLAS_WORKSPACE_CONFIG", "TOKENIZERS_PARALLELISM", "PYTHONPATH")}
                child = subprocess.Popen(stage["argv"], cwd=stage["cwd"], env=env,
                                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                publish("stage-start", current_stage=stage["id"], child_pid=child.pid)
                last_status, last_quota = time.monotonic(), 0
                while child.poll() is None:
                    if requested:
                        raise RuntimeError("Stop signal received")
                    if time.monotonic() - last_quota >= 60:
                        decision = quota()
                        if decision == "CLOSE":
                            raise RuntimeError("Primary quota reached closure threshold")
                        if decision == "STOP_EXPANSION":
                            (run_dir / "STOP_NEW_GPU").touch()
                        last_quota = time.monotonic()
                    if time.monotonic() - last_status >= control["heartbeat_seconds"]:
                        actual = observe(run_dir)
                        publish("heartbeat", actual=actual)
                        require(actual["disk_free_bytes"] >= control.get("runtime_disk_gib", 6) * 1024**3, "Runtime disk reserve reached")
                        last_status = time.monotonic()
                    time.sleep(1)  # Child liveness only; does not read per-episode logs or poll the network.
            entry.update(exit_code=child.returncode, elapsed_seconds=time.monotonic() - started, ended_at=now())
            require(child.returncode == 0, f"Stage {stage['id']} failed: exit {child.returncode}")
            publish("stage-complete")
        else:
            publish("all-stages-complete", state="EXECUTION_COMPLETE", reason="task_commands_complete")
    except BaseException as error:
        # Make the problem visible before backup/closure. stdout is used by the controlling heartbeat.
        print("EXPERIMENT_PROBLEM: " + str(error), flush=True)
        stop_child(child)
        if state["stages"] and "exit_code" not in state["stages"][-1]:
            state["stages"][-1].update(exit_code=child.returncode if child else None, interrupted=True)
        publish("problem", state="INTERRUPTED", reason=str(error))
    finally:
        stop_child(child)
        state["ended_at"] = now()
        state["closure_errors"] = []
        try:
            record_round(repo, config, round_id, state)
        except Exception as error:
            state["closure_errors"].append("report: " + str(error))
        # Preserve a complete local index even if GitHub/auth is unavailable. Never erase data here.
        try:
            run_command(expand(control["backup_argv"], context), repo, run_dir / "backup.log",
                        control.get("backup_timeout_seconds", 3600))
            state["backup_command_ok"] = True
            if (run_dir / "backup-receipt.json").exists():
                state["backup"] = read_json(run_dir / "backup-receipt.json")
        except Exception as error:
            state["closure_errors"].append("backup: " + str(error))
            print("CLOSURE_PROBLEM: backup failed; persistent data retained", flush=True)
        try:
            record_round(repo, config, round_id, state)
        except Exception as error:
            state["closure_errors"].append("report: " + str(error))
        try:
            run_command(expand(control["git_sync_argv"], context), repo, run_dir / "git-sync.log", 300)
            state["git_sync_command_ok"] = True
        except Exception as error:
            state["closure_errors"].append("git: " + str(error))
            print("CLOSURE_PROBLEM: Git sync failed; preserve server disk and report", flush=True)
        # Latest user instruction: an upload/auth problem must still attempt to stop a paid idle server.
        # Provider/host/GPU-idle checks remain mandatory. Denial is recorded, never bypassed.
        try:
            publish("closure-before-shutdown", state="SHUTDOWN_PENDING")
            record_round(repo, config, round_id, state)
        except Exception as error:
            state["closure_errors"].append("final-report: " + str(error))
        try:
            native_shutdown(run_dir, auth["hostname"], execute=True)
        except Exception as error:
            state["closure_errors"].append("shutdown: " + str(error))
            publish("shutdown-blocked", state="SHUTDOWN_BLOCKED")
            print("CLOSURE_PROBLEM: server shutdown NOT confirmed: " + str(error), flush=True)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    return state
