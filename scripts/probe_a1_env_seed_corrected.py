"""No-policy probe for the corrected LIBERO model/environment seed order.

It follows the evaluation reset sequence with fixed task/reset/model seeds and
records raw, preprocessed, and processor-level policy inputs at query 0 and
after eight fixed no-op steps. It is diagnostic only and never scores policy
success or admits a long evaluation by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from experiments.robot.libero.run_libero_eval import prepare_observation
from experiments.robot.openvla_utils import get_processor, normalize_proprio, prepare_images_for_vla
from experiments.robot.robot_utils import get_image_resize_size
from qvla.reproducibility import episode_seeds, seed_all


def sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def snapshot(env, obs, processor, cfg, norm_stats, resize_size, task_description):
    arrays = {
        f"raw/{key}": np.asarray(obs[key]).copy()
        for key in (
            "agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos",
            "robot0_eef_quat", "robot0_gripper_qpos",
        ) if key in obs
    }
    if hasattr(env, "sim"):
        arrays["sim/qpos"] = np.asarray(env.sim.data.qpos).copy()

    policy_obs, _ = prepare_observation(obs, resize_size)
    arrays["policy/full_image_resized"] = np.asarray(policy_obs["full_image"]).copy()
    arrays["policy/wrist_image_resized"] = np.asarray(policy_obs["wrist_image"]).copy()
    arrays["policy/proprio_raw"] = np.asarray(policy_obs["state"]).copy()
    arrays["policy/proprio_normalized"] = normalize_proprio(
        np.asarray(policy_obs["state"]).copy(), norm_stats
    )

    images = prepare_images_for_vla(
        [policy_obs["full_image"], policy_obs["wrist_image"]], cfg
    )
    prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
    primary = processor(prompt, images[0]).to("cpu", dtype=torch.bfloat16)
    wrist = processor(prompt, images[1]).to("cpu", dtype=torch.bfloat16)
    # The evaluator casts processor outputs to BF16 before inference. Convert
    # those already-rounded values to float32 only for portable hashing; each
    # BF16 value is represented exactly by the conversion.
    arrays["model/pixel_values_bf16_as_float32"] = torch.cat(
        [primary["pixel_values"], wrist["pixel_values"]], dim=1
    ).float().cpu().numpy()
    for name in ("input_ids", "attention_mask"):
        if name in primary:
            arrays[f"model/{name}"] = primary[name].cpu().numpy()
    return arrays


def describe(arrays):
    return {
        key: {"sha256": sha(value), "shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in arrays.items()
    }


def compare(left, right):
    result = {}
    for key in sorted(set(left) & set(right)):
        a, b = np.asarray(left[key]), np.asarray(right[key])
        if a.shape != b.shape:
            result[key] = {"shape_changed": True}
            continue
        delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
        result[key] = {
            "same_sha256": sha(a) == sha(b),
            "max_abs": float(delta.max()) if delta.size else 0.0,
            "mean_abs": float(delta.mean()) if delta.size else 0.0,
            "fraction_changed": float(np.count_nonzero(delta) / delta.size) if delta.size else 0.0,
        }
    return result


def reset_snapshot(env, obs):
    """Hash environment-reset state before evaluator fixed-state injection."""
    arrays = {
        f"raw_reset/{key}": np.asarray(obs[key]).copy()
        for key in (
            "agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos",
            "robot0_eef_quat", "robot0_gripper_qpos",
        ) if key in obs
    }
    if hasattr(env, "sim"):
        arrays["sim_reset/qpos"] = np.asarray(env.sim.data.qpos).copy()
        arrays["sim_reset/qvel"] = np.asarray(env.sim.data.qvel).copy()
    return arrays


def capture(task, initial_state, task_id, reset_id, env_seed, processor, cfg,
            norm_stats, resize_size):
    env, description = get_libero_env(task, "openvla", resolution=256)
    try:
        model_seed, derived_env_seed = episode_seeds(
            0, env_seed, "libero_spatial", task_id, reset_id
        )
        # The corrected evaluator order: seed model RNGs first, then LIBERO's
        # process-global NumPy stream immediately before env.reset().
        seed_all(model_seed, strict=False, tensorflow=True)
        env.seed(derived_env_seed)
        reset_obs = env.reset()
        reset_only = reset_snapshot(env, reset_obs)
        obs = env.set_init_state(initial_state)
        pre_solved = bool(env.check_success())
        action = get_libero_dummy_action("openvla")
        wait_done = False
        for _ in range(10):
            obs, _, done, _ = env.step(action)
            wait_done = wait_done or bool(done)
        query0 = snapshot(env, obs, processor, cfg, norm_stats, resize_size, description)
        post_query_done = False
        for _ in range(8):
            obs, _, done, _ = env.step(action)
            post_query_done = post_query_done or bool(done)
        after8 = snapshot(env, obs, processor, cfg, norm_stats, resize_size, description)
        return {
            "derived_model_seed": model_seed,
            "derived_env_seed": derived_env_seed,
            "pre_solved": pre_solved,
            "wait_done": wait_done,
            "post_query_done": post_query_done,
            "reset_before_fixed_init_state_arrays": reset_only,
            "query0_arrays": query0,
            "after8_arrays": after8,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reset", type=int, default=5)
    parser.add_argument("--unnorm-key", default="libero_spatial_no_noops")
    args = parser.parse_args()

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    cfg = SimpleNamespace(
        pretrained_checkpoint=str(args.model_dir),
        center_crop=True,
        model_family="openvla",
    )
    processor = get_processor(cfg)
    stats_path = args.model_dir / "dataset_statistics.json"
    stats_payload = json.loads(stats_path.read_text(encoding="utf-8"))
    norm_stats = stats_payload[args.unnorm_key]["proprio"]
    resize_size = get_image_resize_size(cfg)

    rows = []
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        initial_state = np.asarray(suite.get_task_init_states(task_id)[args.reset])
        captures = {
            str(seed): [
                capture(task, initial_state, task_id, args.reset, seed,
                        processor, cfg, norm_stats, resize_size)
                for _ in range(2)
            ]
            for seed in (0, 1)
        }
        first = {seed: captures[seed][0] for seed in ("0", "1")}
        second = {seed: captures[seed][1] for seed in ("0", "1")}
        rows.append({
            "task_id": task_id,
            "reset_id": args.reset,
            "initial_state_sha256": sha(initial_state),
            "env_seed_0": first["0"]["derived_env_seed"],
            "env_seed_1": first["1"]["derived_env_seed"],
            "task_validity": {
                seed: {
                    "pre_solved": first[seed]["pre_solved"],
                    "wait_done": first[seed]["wait_done"],
                    "post_query_done": first[seed]["post_query_done"],
                }
                for seed in ("0", "1")
            },
            "reset_only_seed_difference": compare(
                first["0"]["reset_before_fixed_init_state_arrays"],
                first["1"]["reset_before_fixed_init_state_arrays"],
            ),
            "within_seed_reset_only_repeat": {
                seed: compare(
                    captures[seed][0]["reset_before_fixed_init_state_arrays"],
                    captures[seed][1]["reset_before_fixed_init_state_arrays"],
                ) for seed in ("0", "1")
            },
            "query0_hashes": {seed: describe(first[seed]["query0_arrays"]) for seed in ("0", "1")},
            "after8_hashes": {seed: describe(first[seed]["after8_arrays"]) for seed in ("0", "1")},
            "query0_seed_difference": compare(first["0"]["query0_arrays"], first["1"]["query0_arrays"]),
            "after8_seed_difference": compare(first["0"]["after8_arrays"], first["1"]["after8_arrays"]),
            "within_seed_repeat": {
                seed: {
                    "query0": compare(first[seed]["query0_arrays"], second[seed]["query0_arrays"]),
                    "after8": compare(first[seed]["after8_arrays"], second[seed]["after8_arrays"]),
                }
                for seed in ("0", "1")
            },
        })

    result = {
        "classification": "Corrected seed-order no-policy diagnostic; not a C0/C3 result or independence claim",
        "changed_variable": "env_seed: 0 -> 1",
        "fixed": {
            "suite": "libero_spatial", "model_seed": 0, "reset_id": args.reset,
            "wait_dummy_steps": 10, "post_query_dummy_steps": 8,
            "seed_call_order": "seed_all(model_seed); env.seed(derived_env_seed); env.reset(); set_init_state",
            "reset_observation_capture": "raw images/state/simulator qpos/qvel before set_init_state, to detect fixed-state masking",
            "policy_inputs": ["two-view processor pixel_values", "input_ids", "attention_mask", "normalized proprio"],
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "tasks": len(rows),
        "query0_model_pixel_differences": sum(
            not row["query0_seed_difference"]["model/pixel_values_bf16_as_float32"]["same_sha256"] for row in rows
        ),
        "query0_repeat_mismatches": sum(
            any(not entry["same_sha256"] for entry in row["within_seed_repeat"]["0"]["query0"].values())
            for row in rows
        ),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
