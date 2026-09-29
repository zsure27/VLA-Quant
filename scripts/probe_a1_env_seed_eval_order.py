"""Audit A1 environment seed with the evaluator's exact seed call order.

This never loads a policy or runs an evaluation episode. It reuses the frozen
LIBERO environment/reset path and applies the same fixed dummy actions for
both environment seeds. The resulting JSON is diagnostic, not a success score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from qvla.reproducibility import episode_seeds, seed_all


OBS_KEYS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)


def capture(task, initial_state: np.ndarray, task_id: int, reset_id: int, env_seed: int):
    env, _ = get_libero_env(task, "openvla", resolution=256)
    try:
        model_seed, derived_env_seed = episode_seeds(0, env_seed, "libero_spatial", task_id, reset_id)
        env.seed(derived_env_seed)
        seed_all(model_seed, strict=False, tensorflow=True)
        env.reset()
        obs = env.set_init_state(initial_state)
        action = get_libero_dummy_action("openvla")
        for _ in range(10):
            obs, _, _, _ = env.step(action)
        first = snapshot(env, obs)
        for _ in range(8):
            obs, _, _, _ = env.step(action)
        return {"derived_env_seed": derived_env_seed, "query0": first, "after_fixed_8": snapshot(env, obs)}
    finally:
        env.close()


def snapshot(env, obs):
    arrays = {key: np.asarray(obs[key]) for key in OBS_KEYS if key in obs}
    if hasattr(env, "sim"):
        arrays["sim_qpos"] = np.asarray(env.sim.data.qpos).copy()
    return {
        key: {
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "value": value.tolist(),
        }
        for key, value in arrays.items()
    }


def compare(a, b):
    result = {}
    for key in sorted(set(a) & set(b)):
        aa, bb = np.asarray(a[key]["value"]), np.asarray(b[key]["value"])
        if aa.shape != bb.shape:
            result[key] = {"shape_changed": True}
            continue
        delta = np.abs(aa.astype(np.float64) - bb.astype(np.float64))
        result[key] = {
            "same_sha256": a[key]["sha256"] == b[key]["sha256"],
            "max_abs": float(delta.max()) if delta.size else 0.0,
            "mean_abs": float(delta.mean()) if delta.size else 0.0,
            "fraction_changed": float(np.count_nonzero(delta) / delta.size) if delta.size else 0.0,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reset", type=int, default=5)
    args = parser.parse_args()
    if args.reset < 5 or args.reset > 49:
        parser.error("A1 probe must remain on development reset 5..49")

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    rows = []
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        initial_state = np.asarray(suite.get_task_init_states(task_id)[args.reset])
        pair = {str(seed): capture(task, initial_state, task_id, args.reset, seed) for seed in (0, 1)}
        rows.append({
            "task_id": task_id,
            "reset_id": args.reset,
            "initial_state_sha256": hashlib.sha256(initial_state.tobytes()).hexdigest(),
            "env_seed_0": pair["0"]["derived_env_seed"],
            "env_seed_1": pair["1"]["derived_env_seed"],
            "query0_difference": compare(pair["0"]["query0"], pair["1"]["query0"]),
            "after_fixed_8_difference": compare(pair["0"]["after_fixed_8"], pair["1"]["after_fixed_8"]),
            "query0_hashes": {seed: {key: value["sha256"] for key, value in p["query0"].items()} for seed, p in pair.items()},
        })

    result = {
        "classification": "A1 post-hoc evaluator-order audit only; no policy or success score",
        "changed_variable": "env_seed: 0 -> 1",
        "fixed": {"suite": "libero_spatial", "model_seed": 0, "reset_id": args.reset,
                  "wait_dummy_steps": 10, "post_query_dummy_steps": 8,
                  "seed_call_order": "env.seed(derived_env_seed), seed_all(model_seed), env.reset(), set_init_state"},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "tasks": len(rows),
                      "changed_query0_qpos": sum(not row["query0_difference"].get("sim_qpos", {}).get("same_sha256", True) for row in rows)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
