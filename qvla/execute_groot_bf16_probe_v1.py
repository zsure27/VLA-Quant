"""One native GR00T/LIBERO BF16 admission probe; never a scored rollout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(args: argparse.Namespace) -> None:
    plan = json.loads(args.plan.resolve(strict=True).read_text())
    if (plan.get("round_id") != "20261107-008-groot-bf16-admission-v1"
            or plan.get("server_id") != "008"
            or plan.get("hostname") != socket.gethostname()
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
            or plan.get("suite") != args.suite
            or plan.get("task_id") != args.task_id
            or plan.get("reset_index") != args.reset_index
            or plan.get("maximum_control_steps") != 10):
        raise ValueError("Admission probe does not match registered plan")
    output = args.output.resolve()
    if str(output) != plan.get("output"):
        raise ValueError("Output differs from registered plan")
    if output.exists():
        raise FileExistsError(output)
    source = args.groot_source.resolve(strict=True)
    libero = args.libero_source.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    receipt = checkpoint / "DOWNLOAD_VERIFICATION.json"
    if not receipt.is_file():
        raise ValueError("Missing verified model snapshot receipt")
    identity = json.loads(receipt.read_text())
    if (identity.get("repo") != "youliangtan/gr00t-n1.5-libero-spatial-posttrain"
            or identity.get("revision") != "03294ae3de7d4870be8ae6e27cbdeffcb46b6425"
            or len(identity.get("files_verified", [])) != identity.get("files")):
        raise ValueError("Unexpected or incomplete Spatial checkpoint receipt")
    if (args.suite, args.task_id, args.reset_index) != ("libero_spatial", 0, 5):
        raise ValueError("Probe identity differs from preregistered case")
    sys.path[:0] = [str(source), str(libero)]
    from examples.Libero.eval.run_libero_eval import GR00TPolicy, get_libero_dummy_action, get_libero_env
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.suite]()
    if suite.n_tasks != 10:
        raise ValueError("Task count mismatch")
    states = suite.get_task_init_states(args.task_id)
    if len(states) != 50:
        raise ValueError("Initial state count mismatch")
    initial = np.asarray(states[args.reset_index])
    if not np.isfinite(initial).all():
        raise ValueError("Nonfinite initial state")
    task = suite.get_task(args.task_id)
    output.mkdir(parents=True)
    report = {
        "gate": "IN_PROGRESS_BF16_NATIVE_PROBE",
        "server_id": "008", "suite": args.suite, "task_id": args.task_id,
        "reset_index": args.reset_index, "initial_state_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
        "checkpoint_receipt_sha256": sha(receipt), "steps_requested": 10,
        "plan_sha256": sha(args.plan),
        "scored_episode": False, "queries_completed": 0,
    }
    (output / "identity.json").write_text(json.dumps(report, indent=2) + "\n")
    t0 = time.perf_counter()
    env = None
    try:
        env, description = get_libero_env(task, resolution=256)
        env.reset()
        obs = env.set_init_state(initial)
        policy = GR00TPolicy(host="localhost", port=args.port, headless=True)
        for _ in range(10):
            obs, _, done, _ = env.step(get_libero_dummy_action())
            if done:
                raise ValueError("Environment ended during stabilization")
        for step in range(10):
            processed = policy._process_observation(obs, task.language)
            arrays = {key: np.asarray(value) for key, value in processed.items() if key != "annotation.human.action.task_description"}
            if any(not np.isfinite(value).all() for value in arrays.values()):
                raise ValueError("Nonfinite policy input")
            raw = policy.policy.get_action(processed)
            action = policy._convert_to_libero_action(raw, 0)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError("Invalid native action")
            sample = output / f"query-{step:03d}.npz"
            np.savez_compressed(sample, **arrays, action=action,
                eef_pos=np.asarray(obs["robot0_eef_pos"]),
                eef_quat=np.asarray(obs["robot0_eef_quat"]),
                gripper_qpos=np.asarray(obs["robot0_gripper_qpos"]))
            obs, _, done, _ = env.step(action.tolist())
            report["queries_completed"] += 1
            if done:
                report["environment_ended"] = True
                break
        report.update(gate="PASS_GROOT_NATIVE_BF16_SHORT_PROBE", task_description=description,
                      wall_seconds=time.perf_counter() - t0,
                      query_files={p.name: sha(p) for p in sorted(output.glob("query-*.npz"))})
    except BaseException as exc:
        report.update(gate="FAIL_GROOT_NATIVE_BF16_SHORT_PROBE", error_type=type(exc).__name__,
                      error=str(exc), wall_seconds=time.perf_counter() - t0)
        raise
    finally:
        if env is not None:
            env.close()
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--groot-source", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--libero-source", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--reset-index", type=int, default=5)
    run(p.parse_args())


if __name__ == "__main__":
    main()
