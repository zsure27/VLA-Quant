"""Run one native GR00T N1.5 LIBERO BF16 episode with per-step trace."""
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


SUITES = {
    "libero_spatial": ("groot_spatial", "youliangtan/gr00t-n1.5-libero-spatial-posttrain", "03294ae3de7d4870be8ae6e27cbdeffcb46b6425", 220),
    "libero_object": ("groot_object", "youliangtan/gr00t-n1.5-libero-object-posttrain", "f422266b016fd917127dd3a0348062a5cd750df43", 280),
    "libero_goal": ("groot_goal", "youliangtan/gr00t-n1.5-libero-goal-posttrain", "17be45f5bf8a543594761b2663c68cf33f33847bd", 600),
    "libero_10": ("groot_long", "youliangtan/gr00t-n1.5-libero-long-posttrain", "aa49078d5cc9ce72917bc43112f1ef12771f277de", 1000),
}
STATES = (5, 15, 25, 35, 45)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_snapshot(checkpoint: Path, suite: str) -> dict:
    name, repo, revision, _ = SUITES[suite]
    if checkpoint.name != name or checkpoint.is_symlink():
        raise ValueError("Wrong or symlinked GR00T checkpoint")
    receipt = checkpoint / "DOWNLOAD_VERIFICATION.json"
    data = json.loads(receipt.read_text())
    rows = data.get("files_verified")
    if data.get("repo") != repo or data.get("revision") != revision or not isinstance(rows, list) or len(rows) != data.get("files"):
        raise ValueError("Checkpoint identity or source receipt differs")
    total = 0
    seen = set()
    for row in rows:
        path = Path(row["path"])
        if path.is_absolute() or ".." in path.parts or path.as_posix() in seen:
            raise ValueError("Unsafe snapshot member")
        seen.add(path.as_posix())
        item = checkpoint / path
        if item.is_symlink() or not item.is_file() or item.stat().st_size != row["bytes"] or sha(item) != row["sha256"]:
            raise ValueError(f"Checkpoint source hash differs: {path}")
        total += row["bytes"]
    if total != data.get("bytes"):
        raise ValueError("Checkpoint total bytes differ")
    return data


def run(args: argparse.Namespace) -> None:
    plan = json.loads(args.plan.resolve(strict=True).read_text())
    if (plan.get("round_id") != args.round_id or plan.get("server_id") != "008"
            or plan.get("hostname") != socket.gethostname()
            or plan.get("suite") != args.suite or plan.get("task_id") != args.task_id
            or plan.get("reset_index") != args.reset_index or plan.get("model_seed") != args.model_seed
            or plan.get("condition") != "bf16" or plan.get("runner") != "qvla.execute_groot_bf16_episode_v1"
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"):
        raise ValueError("Actual argv, GPU or identity differs from registered plan")
    if not 0 <= args.task_id < 10 or args.reset_index not in STATES:
        raise ValueError("Case outside planned task/initial state range")
    output = args.output.resolve()
    if str(output) != plan.get("output") or output.exists():
        raise ValueError("Output differs from plan or already exists")
    import mujoco
    if mujoco.__version__ != "3.3.2":
        raise ValueError("LIBERO reset-verified MuJoCo version differs")
    checkpoint = args.checkpoint.resolve(strict=True)
    source = args.groot_source.resolve(strict=True)
    libero = args.libero_source.resolve(strict=True)
    verified = verify_snapshot(checkpoint, args.suite)
    sys.path[:0] = [str(source), str(libero)]
    from examples.Libero.eval.run_libero_eval import GR00TPolicy, get_libero_dummy_action, get_libero_env, get_libero_image
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.suite]()
    if suite.n_tasks != 10:
        raise ValueError("Unexpected LIBERO task count")
    states = suite.get_task_init_states(args.task_id)
    if len(states) != 50:
        raise ValueError("Unexpected official initial state count")
    initial = np.asarray(states[args.reset_index])
    if not np.isfinite(initial).all():
        raise ValueError("Nonfinite initial state")
    task = suite.get_task(args.task_id)
    output.mkdir(parents=True)
    report = {
        "gate": "IN_PROGRESS_GROOT_BF16_EPISODE", "round_id": args.round_id,
        "server_id": "008", "hostname": socket.gethostname(), "suite": args.suite,
        "task_id": args.task_id, "reset_index": args.reset_index,
        "model_seed": args.model_seed, "condition": "bf16",
        "checkpoint_repo": verified["repo"], "checkpoint_revision": verified["revision"],
        "checkpoint_receipt_sha256": sha(checkpoint / "DOWNLOAD_VERIFICATION.json"),
        "plan_sha256": sha(args.plan), "initial_state_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
        "max_control_steps": SUITES[args.suite][3], "initial_wait_steps": 10,
        "control_steps": 0, "policy_queries": 0, "success": False,
    }
    (output / "identity.json").write_text(json.dumps(report, indent=2) + "\n")
    env = None
    frames = []
    t0 = time.perf_counter()
    try:
        env, description = get_libero_env(task, resolution=256)
        env.reset()
        obs = env.set_init_state(initial)
        for _ in range(10):
            obs, _, done, _ = env.step(get_libero_dummy_action())
            if done:
                raise ValueError("Environment ended during official initial wait")
        policy = GR00TPolicy(host="localhost", port=args.port, headless=True)
        for step in range(SUITES[args.suite][3]):
            top, _wrist = get_libero_image(obs)
            frames.append(top)
            processed = policy._process_observation(obs, task.language)
            arrays = {key: np.asarray(value) for key, value in processed.items()
                      if key != "annotation.human.action.task_description"}
            if any(not np.isfinite(value).all() for value in arrays.values()):
                raise ValueError("Nonfinite policy input")
            raw = policy.policy.get_action(processed)
            action = policy._convert_to_libero_action(raw, 0)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError("Invalid native action")
            before = {name: np.asarray(obs[name]).copy() for name in
                      ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")}
            new_obs, reward, done, info = env.step(action.tolist())
            after = {f"post_{name}": np.asarray(new_obs[name]).copy() for name in before}
            sample = output / f"query-{step:05d}.npz"
            np.savez_compressed(sample, **arrays, action=action, **before, **after,
                                reward=np.asarray(reward), done=np.asarray(done))
            report["control_steps"] += 1
            report["policy_queries"] += 1
            obs = new_obs
            if done:
                report["success"] = True
                break
        if report["control_steps"] < 1 or report["policy_queries"] != report["control_steps"]:
            raise ValueError("No complete native control trajectory")
        import imageio.v2 as imageio
        video = output / "rollout.mp4"
        imageio.mimsave(video, frames, fps=20)
        report.update(gate="PASS_GROOT_NATIVE_BF16_EPISODE", task_description=description,
                      video_sha256=sha(video), query_files={p.name: sha(p) for p in sorted(output.glob("query-*.npz"))})
    except BaseException as exc:
        report.update(gate="FAIL_GROOT_NATIVE_BF16_EPISODE", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if env is not None:
            env.close()
        report["wall_s_including_env_and_policy"] = time.perf_counter() - t0
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--groot-source", type=Path, required=True)
    p.add_argument("--libero-source", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--round-id", required=True)
    p.add_argument("--suite", choices=SUITES, required=True)
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--reset-index", type=int, required=True)
    p.add_argument("--model-seed", type=int, default=7)
    p.add_argument("--port", type=int, default=5555)
    run(p.parse_args())


if __name__ == "__main__":
    main()
