"""Check native JAX and converted PyTorch pi0.5 on one identical LIBERO query.

This is a numerical admission probe, not a quantization result. The JAX probe
reconstructs the native policy's first noise draw from RNG key zero. Run JAX
and PyTorch in separate processes to avoid keeping two models in GPU memory.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise ValueError("Unique bridge output already exists")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.openpi_source, text=True).strip() != OPENPI_COMMIT:
        raise ValueError("OpenPI source commit differs")
    if args.backend == "torch" and args.jax_reference is None:
        raise ValueError("Torch bridge requires the completed JAX reference")
    args.output.mkdir(parents=True)
    report = {
        "gate": "IN_PROGRESS_PI05_SAME_INPUT_BRIDGE",
        "backend": args.backend,
        "query_sha256": sha(args.query),
        "checkpoint": str(args.checkpoint.resolve(strict=True)),
        "openpi_commit": OPENPI_COMMIT,
    }
    try:
        with np.load(args.query, allow_pickle=False) as query:
            required = {"image", "wrist_image", "state", "prompt", "raw_policy_chunk"}
            if not required.issubset(query.files):
                raise ValueError("Incomplete native query")
            obs = {
                "observation/image": query["image"],
                "observation/wrist_image": query["wrist_image"],
                "observation/state": query["state"],
                "prompt": str(query["prompt"].item()),
            }
            original = np.asarray(query["raw_policy_chunk"])
        sys.path[:0] = [str(args.openpi_source), str(args.openpi_source / "src"),
                        str(args.openpi_source / "packages/openpi-client/src")]
        import jax
        import jax.numpy as jnp
        from openpi.training import config as training_config
        from openpi.training import checkpoints
        from openpi.policies import policy_config

        config = training_config.get_config("pi05_libero")
        if args.backend == "jax":
            _, sample_key = jax.random.split(jax.random.key(0))
            noise = np.asarray(jax.random.normal(sample_key, (1, config.model.action_horizon,
                                                              config.model.action_dim), dtype=jnp.float32))[0]
            policy = policy_config.create_trained_policy(config, args.checkpoint)
            reference = np.asarray(policy.infer(obs)["actions"])
            report["prior_device_max_abs"] = float(np.max(np.abs(reference.astype(np.float64) - original.astype(np.float64))))
        else:
            with np.load(args.jax_reference / "noise.npz", allow_pickle=False) as saved:
                noise = np.asarray(saved["noise"])
            with np.load(args.jax_reference / "actions.npz", allow_pickle=False) as saved:
                reference = np.asarray(saved["actions"])
            with (args.jax_reference / "result.json").open(encoding="utf-8") as stream:
                jax_report = json.load(stream)
            if jax_report.get("gate") != "PASS_PI05_SAME_DEVICE_JAX_NOISE_BRIDGE" or jax_report["query_sha256"] != report["query_sha256"]:
                raise ValueError("JAX replay has not passed on this exact query")
            config = dataclasses.replace(config, model=dataclasses.replace(config.model, pytorch_compile_mode=None))
            data_config = config.data.create(config.assets_dirs, config.model)
            norm_stats = checkpoints.load_norm_stats(args.native_checkpoint / "assets", data_config.asset_id)
            policy = policy_config.create_trained_policy(config, args.checkpoint,
                                                         norm_stats=norm_stats, pytorch_device="cuda:0")
        if noise.shape != (config.model.action_horizon, config.model.action_dim) or not np.isfinite(noise).all() or original.shape != (config.model.action_horizon, 7):
            raise ValueError("Unexpected native pi0.5 noise or action shape")
        actions = np.asarray(policy.infer(obs, noise=noise)["actions"])
        if actions.shape != reference.shape or not np.isfinite(actions).all():
            raise ValueError("Invalid bridge actions")
        np.savez_compressed(args.output / "noise.npz", noise=noise)
        np.savez_compressed(args.output / "actions.npz", actions=actions)
        error = np.asarray(actions, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
        report.update(
            max_abs=float(np.max(np.abs(error))),
            rmse=float(np.sqrt(np.mean(error ** 2))),
            exact_equal=bool(np.array_equal(actions, reference)),
            noise_sha256=sha(args.output / "noise.npz"),
            actions_sha256=sha(args.output / "actions.npz"),
        )
        if args.backend == "jax":
            report["gate"] = "PASS_PI05_SAME_DEVICE_JAX_NOISE_BRIDGE" if report["exact_equal"] else "FAIL_PI05_SAME_DEVICE_JAX_NOISE_BRIDGE"
        else:
            report["jax_reference_sha256"] = sha(args.jax_reference / "result.json")
            report["gate"] = ("PASS_PI05_SAME_INPUT_TORCH_BRIDGE"
                              if report["max_abs"] <= 0.02 and report["rmse"] <= 0.002
                              else "FAIL_PI05_SAME_INPUT_TORCH_BRIDGE")
    except BaseException as error:
        report.update(gate="ERROR_PI05_SAME_INPUT_BRIDGE", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not report["gate"].startswith("PASS_"):
        raise RuntimeError(report["gate"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", required=True, choices=("jax", "torch"))
    for name in ("query", "openpi_source", "checkpoint", "native_checkpoint", "output"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--jax-reference", type=Path)
    run(p.parse_args())


if __name__ == "__main__":
    main()
