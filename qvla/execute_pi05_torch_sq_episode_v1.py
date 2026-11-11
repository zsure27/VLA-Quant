"""Independent official PyTorch BF16 trajectories; separate from native JAX cohort."""
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


SUITES = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
STATES = (0, 5, 15, 25, 35, 45)
SOURCE = "gs://openpi-assets/checkpoints/pi05_libero/"
OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_head(path: Path) -> str:
    import subprocess
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def verify_model(root: Path) -> dict:
    receipt_path = root / "DOWNLOAD_VERIFICATION.json"
    receipt = json.loads(receipt_path.read_text())
    rows = receipt.get("files")
    if receipt.get("source") != SOURCE or receipt.get("objects") != 16 or len(rows) != 16:
        raise ValueError("Wrong or incomplete public checkpoint receipt")
    total = 0
    for row in rows:
        rel = Path(row["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe checkpoint member")
        file = root / rel
        if file.is_symlink() or not file.is_file() or file.stat().st_size != row["bytes"] or sha(file) != row["sha256"]:
            raise ValueError(f"Checkpoint member differs: {rel}")
        total += row["bytes"]
    if total != receipt["bytes"]:
        raise ValueError("Checkpoint total differs")
    return receipt


def run(args: argparse.Namespace) -> None:
    total_started = time.perf_counter()
    plan = json.loads(args.plan.resolve(strict=True).read_text())
    output = args.output.resolve()
    if (plan.get("round_id") != args.round_id or plan.get("server_id") != "084"
            or plan.get("hostname") != socket.gethostname() or plan.get("runner") != "qvla.execute_pi05_torch_sq_episode_v1"
            or plan.get("suite") != args.suite or plan.get("task_id") != args.task_id
            or plan.get("reset_index") != args.reset_index or plan.get("model_seed") != args.model_seed
            or plan.get("condition") != f"smoothquant_w4a{args.activation_bits}" or plan.get("output") != str(output)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or output.exists()):
        raise ValueError("Execution differs from unique preregistered plan")
    if args.suite not in SUITES or args.task_id not in range(10) or args.reset_index not in STATES:
        raise ValueError("Case outside registered study range")
    model = args.checkpoint.resolve(strict=True)
    openpi = args.openpi_source.resolve(strict=True)
    libero_source = args.libero_source.resolve(strict=True)
    if git_head(openpi) != OPENPI_COMMIT or git_head(libero_source) != LIBERO_COMMIT:
        raise ValueError("Unpinned upstream source")
    from qvla.pi05_torch_backend_v1 import BACKEND_ID, load_policy, noise_stream, next_noise
    admission = json.loads(Path(plan["admission_path"]).read_text())
    if (sha(Path(plan["admission_path"])) != plan["admission_sha256"]
            or admission["gate"] != "PASS_PI05_TORCH_INDEPENDENT_BF16_RELOAD_INPUT"
            or plan.get("backend") != BACKEND_ID
            or sha(model / "model.safetensors") != admission["checkpoint_sha256"]):
        raise ValueError("Independent Torch BF16 admission/content mismatch")
    if args.reset_index == 0 and plan.get("role") != "calibration":
        raise ValueError("Reset0 is reserved for calibration")
    if args.reset_index != 0 and plan.get("role") != "development":
        raise ValueError("Evaluation role differs")
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_config.resolve(strict=True))
    sys.path[:0] = [str(openpi), str(openpi / "src"), str(openpi / "packages/openpi-client/src"), str(libero_source)]
    import torch
    from libero.libero import benchmark
    from openpi.training import config as training_config
    from openpi.policies import policy_config
    from openpi_client import image_tools
    from examples.libero.main import _get_libero_env, _quat2axisangle, LIBERO_DUMMY_ACTION

    # Only the pinned LIBERO initial-state file may use legacy pickle loading.
    init_root = (libero_source / "libero/libero/init_files").resolve(strict=True)
    original_load = torch.load

    def load_trusted_init(file, *a, **kw):
        path = Path(file).resolve(strict=True) if isinstance(file, (str, Path)) else None
        if path is not None and path.is_relative_to(init_root) and "weights_only" not in kw:
            kw["weights_only"] = False
        return original_load(file, *a, **kw)

    torch.load = load_trusted_init
    try:
        suite = benchmark.get_benchmark_dict()[args.suite]()
        if suite.n_tasks != 10:
            raise ValueError("Unexpected suite task count")
        states = suite.get_task_init_states(args.task_id)
    finally:
        torch.load = original_load
    if len(states) != 50:
        raise ValueError("Unexpected official initial state count")
    initial = np.asarray(states[args.reset_index])
    if not np.isfinite(initial).all():
        raise ValueError("Nonfinite initial state")
    task = suite.get_task(args.task_id)
    config = training_config.get_config("pi05_libero")
    policy = load_policy(model.parent.parent)
    from qvla.pi05_smoothquant_v1 import apply
    if sha(args.profile) != plan["profile_sha256"]:
        raise ValueError("Profile content differs from the preregistered plan")
    payload = torch.load(args.profile, map_location="cpu", weights_only=True)
    handles = apply(policy._model, payload, model.parent.parent, args.activation_bits)
    if len(handles) != 414:
        raise ValueError("Not all action-connected targets entered SQ")
    probe = payload["calibration_probe_samples"][0]
    if sha(Path(probe["path"])) != probe["sha256"]:
        raise ValueError("Profile replay observation content mismatch")
    with np.load(probe["path"], allow_pickle=False) as q:
        probe_input = {"observation/image": q["image"], "observation/wrist_image": q["wrist_image"],
                       "observation/state": q["state"], "prompt": str(q["prompt"].item())}
        probe_noise = q["policy_noise"]
    actual_probe = np.asarray(policy.infer(probe_input, noise=probe_noise)["actions"])
    if not np.array_equal(actual_probe, payload["calibration_expected_actions"][str(args.activation_bits)][0].numpy()):
        raise ValueError("Independent-process SQ profile replay action mismatch")
    rng = noise_stream()
    output.mkdir(parents=True)
    report = {"gate": "IN_PROGRESS_PI05_TORCH_SQ_EPISODE", "round_id": args.round_id,
              "server_id": "084", "hostname": socket.gethostname(), "condition": f"smoothquant_w4a{args.activation_bits}",
              "profile_sha256": sha(args.profile), "target_count": 414, "weight_bits": 4, "activation_bits": args.activation_bits,
              "suite": args.suite, "task_id": args.task_id, "reset_index": args.reset_index,
              "model_seed": args.model_seed, "openpi_commit": OPENPI_COMMIT,
              "libero_commit": LIBERO_COMMIT, "checkpoint_sha256": sha(model / "model.safetensors"),
              "backend": BACKEND_ID, "role": plan["role"], "cohort_id": plan["cohort_id"],
              "plan_sha256": sha(args.plan), "initial_state_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
              "max_control_steps": SUITES[args.suite], "initial_wait_steps": 10,
              "replan_steps": 5, "control_steps": 0, "policy_queries": 0, "success": False}
    (output / "identity.json").write_text(json.dumps(report, indent=2) + "\n")
    env = None
    frames = []
    t0 = time.perf_counter()
    try:
        env, description = _get_libero_env(task, 256, args.model_seed)
        env.reset()
        obs = env.set_init_state(initial)
        for _ in range(10):
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            if done:
                raise ValueError("Episode ended during native wait")
        action_plan = []
        for step in range(SUITES[args.suite]):
            image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
            wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))
            frames.append(image)
            before = {name: np.asarray(obs[name]).copy() for name in
                      ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")}
            if not action_plan:
                state = np.concatenate((before["robot0_eef_pos"],
                                        _quat2axisangle(before["robot0_eef_quat"].copy()),
                                        before["robot0_gripper_qpos"]))
                inputs = {"observation/image": image, "observation/wrist_image": wrist,
                          "observation/state": state, "prompt": str(description)}
                policy_noise = next_noise(rng)
                if report["policy_queries"] == 0:
                    with np.load(Path(plan["bf16_reference"]) / "query-00000.npz", allow_pickle=False) as baseline:
                        checks = {"image": np.array_equal(image, baseline["image"]),
                                  "wrist_image": np.array_equal(wrist, baseline["wrist_image"]),
                                  "state": np.array_equal(state, baseline["state"]),
                                  "prompt": str(description) == str(baseline["prompt"].item()),
                                  "policy_noise": np.array_equal(policy_noise, baseline["policy_noise"])}
                    (output / "first-query-checks.json").write_text(json.dumps(checks, indent=2))
                    if not all(checks.values()):
                        np.savez_compressed(output / "first-query-failed-input.npz", image=image,wrist_image=wrist,
                                            state=state,prompt=np.asarray(description),policy_noise=policy_noise)
                        raise ValueError("Actual first observation/noise differs from same-backend BF16 reference")
                actions = np.asarray(policy.infer(inputs, noise=policy_noise)["actions"])
                if actions.ndim != 2 or actions.shape[0] < 5 or actions.shape[1] != 7 or not np.isfinite(actions).all():
                    raise ValueError("Invalid native pi0.5 action chunk")
                query_path = output / f"query-{report['policy_queries']:05d}.npz"
                np.savez_compressed(query_path, image=image, wrist_image=wrist, state=state,
                                    prompt=np.asarray(description), raw_policy_chunk=actions, policy_noise=policy_noise,
                                    control_step=np.asarray(step), **before)
                report["policy_queries"] += 1
                action_plan = list(actions[:5])
            action = np.asarray(action_plan.pop(0))
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError("Invalid executed action")
            new_obs, reward, done, info = env.step(action.tolist())
            after = {f"post_{name}": np.asarray(new_obs[name]).copy() for name in before}
            np.savez_compressed(output / f"step-{step:05d}.npz", action=action,
                                reward=np.asarray(reward), done=np.asarray(done), **before, **after)
            report["control_steps"] += 1
            obs = new_obs
            if done:
                report["success"] = True
                break
        if report["control_steps"] < 1 or report["policy_queries"] < 1:
            raise ValueError("No complete native trajectory")
        import imageio.v2 as imageio
        video = output / "rollout.mp4"
        imageio.mimsave(video, frames, fps=20, codec="libx264")
        report.update(gate="PASS_PI05_TORCH_SQ_EPISODE", task_description=description,
                      video_sha256=sha(video),
                      query_files={p.name: sha(p) for p in sorted(output.glob("query-*.npz"))},
                      step_files={p.name: sha(p) for p in sorted(output.glob("step-*.npz"))})
    except BaseException as exc:
        report.update(gate="FAIL_PI05_TORCH_SQ_EPISODE", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if env is not None:
            env.close()
        report["wall_s_env_and_queries"] = time.perf_counter() - t0
        report["wall_s_including_verification_and_load"] = time.perf_counter() - total_started
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "openpi_source", "libero_source", "libero_config", "plan", "output"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--activation-bits", type=int, choices=[4, 8], required=True)
    p.add_argument("--round-id", required=True)
    p.add_argument("--suite", choices=SUITES, required=True)
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--reset-index", type=int, required=True)
    p.add_argument("--model-seed", type=int, default=7)
    run(p.parse_args())


if __name__ == "__main__":
    main()
