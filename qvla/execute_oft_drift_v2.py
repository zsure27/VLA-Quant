"""Trace one native OpenVLA-OFT LIBERO episode without changing policy actions.

This entry point uses the pinned upstream evaluator and the existing read-only
episode observer. It is deliberately one case per invocation until the actual
model, simulator and trace contracts have passed on the current machine.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from qvla.trajectory_capture_v2 import observe_episode


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
STATES = (5, 15, 25, 35, 45)
CHECKPOINT_REPO = "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
CHECKPOINT_REVISION = "638918f3d1c2e43a39a8a20772bdb8b91835e4b7"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_identity(root: Path) -> str:
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def verify_checkpoint_receipt(checkpoint: Path) -> dict:
    """Recheck the pinned source hashes; older download receipts omit ``gate``."""
    receipt = checkpoint / "DOWNLOAD_VERIFICATION.json"
    if not receipt.is_file() or receipt.is_symlink():
        raise ValueError("Full pinned checkpoint verification receipt required")
    verified = json.loads(receipt.read_text())
    if verified.get("repo") != CHECKPOINT_REPO or verified.get("revision") != CHECKPOINT_REVISION:
        raise ValueError("Unexpected checkpoint identity")
    if verified.get("gate") not in (None, "PASS_PUBLIC_MODEL_SNAPSHOT_HASHES"):
        raise ValueError("Checkpoint verification gate mismatch")
    rows = verified.get("files_verified")
    if not isinstance(rows, list) or not rows or len(rows) != verified.get("files"):
        raise ValueError("Incomplete checkpoint verification")
    total = 0
    names = set()
    for row in rows:
        name = Path(row["path"])
        if name.is_absolute() or ".." in name.parts or not name.parts or name.as_posix() in names:
            raise ValueError("Unsafe or duplicate checkpoint member")
        names.add(name.as_posix())
        file = checkpoint / name
        size = row["bytes"]
        if not isinstance(size, int) or size < 0 or file.is_symlink() or not file.is_file():
            raise ValueError("Missing checkpoint member")
        if file.stat().st_size != size:
            raise ValueError("Checkpoint member byte count mismatch")
        actual_sha256 = sha256_file(file)
        if actual_sha256 != row["sha256"]:
            raise ValueError("Checkpoint member SHA256 mismatch")
        lfs_sha256 = row.get("source_lfs_sha256")
        if lfs_sha256:
            if actual_sha256 != lfs_sha256:
                raise ValueError("Checkpoint member source LFS SHA256 mismatch")
        else:
            expected_blob = row.get("source_git_blob")
            if not isinstance(expected_blob, str):
                raise ValueError("Checkpoint member lacks source hash")
            blob = hashlib.sha1(f"blob {size}\0".encode())
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    blob.update(block)
            if blob.hexdigest() != expected_blob:
                raise ValueError("Checkpoint member source Git blob mismatch")
        total += size
    if total != verified.get("bytes"):
        raise ValueError("Checkpoint snapshot byte count mismatch")
    return verified


def run(args: argparse.Namespace) -> None:
    source = args.oft_source.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    verified = verify_checkpoint_receipt(checkpoint)
    if not (0 <= args.task_id < 10) or args.initial_state_index not in STATES or args.suite not in SUITES:
        raise ValueError("Case is outside preregistered task or initial states")
    if os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "", "-1"):
        raise ValueError("Explicit GPU selection required")
    sys.path.insert(0, str(source))
    module = importlib.import_module("experiments.robot.libero.run_libero_eval")
    cfg = module.GenerateConfig(
        pretrained_checkpoint=str(checkpoint), task_suite_name=args.suite,
        num_trials_per_task=1, use_wandb=False, seed=args.model_seed,
        num_images_in_input=2, use_proprio=True, use_l1_regression=True,
        num_open_loop_steps=8, initial_states_path="DEFAULT",
    )
    cfg.seed_protocol = "paired"
    module.validate_config(cfg)
    module.set_seed_everywhere(cfg.seed)
    t0 = time.perf_counter()
    model, action_head, proprio_projector, noisy_projector, processor = module.initialize_model(cfg)
    resize_size = module.get_image_resize_size(cfg)
    suite = module.benchmark.get_benchmark_dict()[args.suite]()
    if suite.n_tasks != 10:
        raise ValueError("Unexpected LIBERO task count")
    task = suite.get_task(args.task_id)
    states = suite.get_task_init_states(args.task_id)
    if len(states) != 50:
        raise ValueError("Unexpected official initial state count")
    initial_state = np.asarray(states[args.initial_state_index])
    if not np.isfinite(initial_state).all():
        raise ValueError("Non-finite initial state")
    state_sha = hashlib.sha256(initial_state.tobytes()).hexdigest()
    env, description = module.get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    output.mkdir(parents=True)
    identity = {
        "study": "multisuite-trajectory-deviation-v1",
        "round_id": args.round_id,
        "model": "openvla_oft",
        "condition": "bf16",
        "checkpoint_repo": verified["repo"],
        "checkpoint_revision": verified["revision"],
        "checkpoint_receipt_sha256": sha256_file(checkpoint / "DOWNLOAD_VERIFICATION.json"),
        "oft_source_commit": source_identity(source),
        "task_description": description,
    }
    seed = args.model_seed
    try:
        with observe_episode(module, output / "episodes", identity):
            module.log_message("EPISODE_MANIFEST " + json.dumps({
                "task_id": args.task_id,
                "init_state_index": args.initial_state_index,
                "model_seed": seed,
                "env_seed": seed,
                "init_state_sha256": state_sha,
            }))
            success, images = module.run_episode(
                cfg, env, description, model, resize_size,
                processor=processor, action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_projector,
                initial_state=initial_state,
            )
    finally:
        env.close()
    episode_dir = output / "episodes" / f"task-{args.task_id:02d}-reset-{args.initial_state_index:03d}"
    episode = json.loads((episode_dir / "episode.json").read_text())
    if episode["status"] not in ("success", "timeout") or bool(success) != (episode["status"] == "success"):
        raise ValueError("Episode outcome/trace mismatch")
    if episode["control_steps"] < 1 or episode["policy_queries"] < 1:
        raise ValueError("No admitted natural control trajectory")
    # Video is an auxiliary replay of actual evaluator frames, not a source of
    # pose measurements. Its upload belongs in the private archive.
    import imageio.v2 as imageio

    video = output / "rollout.mp4"
    imageio.mimsave(video, images, fps=20)
    report = {
        **identity,
        "suite": args.suite,
        "task_id": args.task_id,
        "initial_state_index": args.initial_state_index,
        "initial_state_sha256": state_sha,
        "success": bool(success),
        "control_steps": episode["control_steps"],
        "policy_queries": episode["policy_queries"],
        "wall_s_including_load": time.perf_counter() - t0,
        "video_sha256": sha256_file(video),
        "gate": "PASS_ACTUAL_OFT_BF16_NATURAL_TRACE",
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oft-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--initial-state-index", type=int, required=True)
    parser.add_argument("--model-seed", type=int, default=7)
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
