"""No-model CPU observation bridge through the real LIBERO evaluator.

This does not evaluate BF16/AWQ success, reproduce model-loading RNG use, or
admit GPU experiments. Historical sources and outputs are never modified.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "results/shared/source-snapshots/oft-libero-evaluator-fe37e8097c28.py"
HISTORICAL_SHA = "fe37e8097c286e1946e5194a1f817a2b2362d167a6a5a49443c5aa20314d8873"
PROTOCOLS = ("upstream_stream", "historical_paired", "corrected_paired")
FIXED_RAW_ACTION = (0., 0., 0., 0., 0., 0., 1.)
RAW_KEYS = ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos",
            "robot0_eef_quat", "robot0_gripper_qpos")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_seed_order(path):
    """Check the actual run_task function, rather than matching comments."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    task = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_task"), None)
    if task is None:
        raise ValueError("Evaluator has no run_task function")
    calls = []
    for node in ast.walk(task):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                and func.value.id == "env" and func.attr == "seed"):
            calls.append((node.lineno, "env.seed"))
        elif isinstance(func, ast.Name) and func.id == "seed_all":
            calls.append((node.lineno, "seed_all"))
    order = [name for _, name in sorted(calls)]
    if len(order) != 2 or set(order) != {"env.seed", "seed_all"}:
        raise ValueError(f"Unrecognized evaluator paired seed sequence: {order}")
    return order


@contextmanager
def patched_attributes(obj, changes):
    originals = {name: getattr(obj, name) for name in changes}
    try:
        for name, value in changes.items():
            setattr(obj, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(obj, name, value)


def plan(args):
    if not 1 <= args.tasks <= 10 or not 1 <= args.resets <= 2 or args.repeats != 2:
        raise ValueError("Bounded probe requires 1..10 tasks, 1..2 consecutive resets and exactly two repeats")
    if args.env_seed != 0 or not 0 <= args.model_seed < 2**32:
        raise ValueError("Upstream bridge requires env_seed=0 and a valid model seed")
    if not args.historical_evaluator.is_file() or sha(args.historical_evaluator) != args.historical_sha:
        raise ValueError("Historical evaluator content does not match its required SHA")
    if inspect_seed_order(args.historical_evaluator) != ["env.seed", "seed_all"]:
        raise ValueError("Historical paired order is not the archived env.seed then seed_all")
    return {
        "classification": "CPU no-model fixed-action environment/preprocessing diagnostic",
        "protocols": list(PROTOCOLS), "tasks": args.tasks, "reset_indices": list(range(args.resets)),
        "repeats": args.repeats, "model_seed": args.model_seed, "env_seed": args.env_seed,
        "episodes": len(PROTOCOLS) * args.tasks * args.resets * args.repeats,
        "queries_per_episode": 2, "wait_steps": 10, "fixed_execution_steps": 16,
        "chunk": [8, 7], "historical_evaluator_sha256": args.historical_sha,
        "limits": [
            "Upstream environment/preprocessing stream only; 7B model initialization and forward RNG are excluded.",
            "Fixed sixteen-step episode length differs from real success-dependent rollout lengths.",
            "No success-rate, quantization, independence or long-evaluation admission claim.",
            "Historical seed order is bound to the archived evaluator; current utility sources are separately recorded.",
        ],
    }


def compare_runs(runs):
    expected = set(runs[0]["episodes"])
    if not expected or any(set(run["episodes"]) != expected for run in runs):
        raise ValueError("Probe episode coverage differs between protocols or repeats")
    within = {}
    for protocol in PROTOCOLS:
        pair = [r for r in runs if r["protocol"] == protocol]
        if len(pair) != 2:
            raise ValueError("Each protocol must have exactly two independently seeded repeats")
        mismatch = [key for key in sorted(expected)
                    if pair[0]["episodes"][key] != pair[1]["episodes"][key]]
        within[protocol] = {"repeat_equal": not mismatch, "mismatch_episode_keys": mismatch}
    cross = {}
    for left, right in ((PROTOCOLS[0], PROTOCOLS[1]), (PROTOCOLS[0], PROTOCOLS[2]), (PROTOCOLS[1], PROTOCOLS[2])):
        a = next(r for r in runs if r["protocol"] == left)["episodes"]
        b = next(r for r in runs if r["protocol"] == right)["episodes"]
        differences = {}
        for key in sorted(expected):
            changed = {}
            for query in ("query0", "after_fixed_8"):
                aa, bb = a[key]["queries"][query], b[key]["queries"][query]
                changed[query] = sorted(k for k in set(aa) | set(bb) if aa.get(k) != bb.get(k))
            if any(changed.values()):
                differences[key] = changed
        cross[f"{left}__vs__{right}"] = {"changed_episodes": len(differences), "differences": differences}
    return within, cross


def run(args, specification):
    # Set before importing torch/TensorFlow/LIBERO. This program must not claim a GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    for root in (args.oft_root, args.libero_root):
        if root is not None:
            sys.path.insert(0, str(root.resolve()))
    import numpy as np
    import torch
    import tensorflow as tf
    torch.set_num_threads(1)
    tf.config.set_visible_devices([], "GPU")
    if torch.cuda.is_available() or tf.config.get_visible_devices("GPU"):
        raise RuntimeError("CPU bridge refuses a visible GPU")
    from libero.libero import benchmark
    import experiments.robot.libero.run_libero_eval as current
    import experiments.robot.openvla_utils as U
    from qvla.reproducibility import seed_all
    from transformers import AutoProcessor

    current_path = Path(current.__file__).resolve()
    if inspect_seed_order(current_path) != ["seed_all", "env.seed"]:
        raise RuntimeError("Current evaluator is not the corrected paired-order version")
    spec = importlib.util.spec_from_file_location("qvla_historical_bridge_evaluator", args.historical_evaluator)
    historical = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = historical
    spec.loader.exec_module(historical)
    # Local processor only: no model weights, AutoModel or network downloads.
    processor = AutoProcessor.from_pretrained(str(args.model_dir), trust_remote_code=True, local_files_only=True)
    statistics_path = args.model_dir / "dataset_statistics.json"
    stats = json.loads(statistics_path.read_text(encoding="utf-8"))
    key = args.unnorm_key or ("libero_spatial" if "libero_spatial" in stats else "libero_spatial_no_noops")
    if key not in stats or "proprio" not in stats[key]:
        raise ValueError("Requested proprio normalization statistics are absent")
    norm = stats[key]["proprio"]
    started = time.monotonic()
    runs = []
    sources = {"current_evaluator": {"path": str(current_path), "sha256": sha(current_path)},
               "historical_evaluator": {"path": str(args.historical_evaluator), "sha256": sha(args.historical_evaluator)},
               "openvla_utils": {"path": str(Path(U.__file__).resolve()), "sha256": sha(U.__file__)},
               "statistics": {"path": str(statistics_path), "sha256": sha(statistics_path)}}
    from experiments.robot.libero import libero_utils
    sources["libero_utils"] = {"path": str(Path(libero_utils.__file__).resolve()), "sha256": sha(libero_utils.__file__)}

    def describe(array):
        value = np.ascontiguousarray(array)
        return {"sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                "shape": list(value.shape), "dtype": str(value.dtype)}

    for protocol in PROTOCOLS:
        evaluator = historical if protocol == "historical_paired" else current
        for repeat in range(2):
            cfg = evaluator.GenerateConfig(pretrained_checkpoint=str(args.model_dir), task_suite_name="libero_spatial",
                initial_state_offset=0, num_trials_per_task=args.resets, initial_states_path="DEFAULT",
                seed=args.model_seed, env_seed=0, seed_protocol="upstream" if protocol == "upstream_stream" else "paired",
                num_steps_wait=10, num_open_loop_steps=8, num_images_in_input=2, use_proprio=True,
                use_l1_regression=True, use_diffusion=False, center_crop=True)
            cfg.unnorm_key = key
            evaluator.validate_config(cfg)
            evaluator.set_seed_everywhere(cfg.seed)
            if cfg.seed_protocol == "paired":
                seed_all(cfg.seed, strict=False, tensorflow=True)
            suite = benchmark.get_benchmark_dict()["libero_spatial"]()
            if suite.n_tasks != 10 or evaluator.NUM_ACTIONS_CHUNK != 8:
                raise RuntimeError("Spatial task/chunk contract changed")
            episodes = {}
            state = {"manifest": None, "active": None, "raw": None, "env": None}
            original_episode, original_prepare, original_env = evaluator.run_episode, evaluator.prepare_observation, evaluator.get_libero_env
            dummy = np.asarray(libero_utils.get_libero_dummy_action("openvla"), dtype=np.float64)
            transformed = np.asarray(evaluator.process_action(np.array(FIXED_RAW_ACTION), "openvla"))
            if dummy.shape != (7,) or not np.array_equal(dummy, transformed):
                raise RuntimeError("Fixed raw chunk is no longer equivalent to the evaluator dummy action")

            class EnvProxy:
                def __init__(self, env):
                    self.env, self.steps = env, 0
                def __getattr__(self, name):
                    return getattr(self.env, name)
                def reset(self, *a, **kw):
                    self.steps = 0
                    return self.env.reset(*a, **kw)
                def set_init_state(self, *a, **kw):
                    observation = self.env.set_init_state(*a, **kw)
                    if callable(getattr(self.env, "check_success", None)) and self.env.check_success():
                        raise RuntimeError("Official initial state is already solved")
                    return observation
                def step(self, action):
                    if not np.array_equal(np.asarray(action), dummy):
                        raise RuntimeError("Probe attempted a non-fixed action")
                    result = self.env.step(action)
                    self.steps += 1
                    if result[2]:
                        raise RuntimeError("Fixed-action probe unexpectedly terminated successfully")
                    return result

            def new_env(*a, **kw):
                environment, instruction = original_env(*a, **kw)
                return EnvProxy(environment), instruction

            def log(message, log_file=None):
                message = str(message)
                if message.startswith("EPISODE_MANIFEST "):
                    state["manifest"] = json.loads(message[len("EPISODE_MANIFEST "):])
                elif message.startswith("Episode error:"):
                    raise RuntimeError(message)

            def prepare(obs, resize_size):
                state["raw"] = {f"raw/{k}": np.asarray(obs[k]).copy() for k in RAW_KEYS if k in obs}
                return original_prepare(obs, resize_size)

            def fixed_action(_cfg, model, obs, instruction, **kwargs):
                active = state["active"]
                if active is None or state["raw"] is None:
                    raise RuntimeError("Policy query occurred outside the real episode/preprocessing chain")
                label = "query0" if not active["queries"] else "after_fixed_8"
                if label in active["queries"]:
                    raise RuntimeError("Unexpected third query in bounded probe")
                arrays = dict(state["raw"])
                environment = state["env"]
                if hasattr(environment, "sim"):
                    arrays["sim/qpos"] = np.asarray(environment.sim.data.qpos).copy()
                    arrays["sim/qvel"] = np.asarray(environment.sim.data.qvel).copy()
                arrays.update({"policy/full_image": np.asarray(obs["full_image"]),
                               "policy/wrist_image": np.asarray(obs["wrist_image"]),
                               "policy/proprio_raw": np.asarray(obs["state"]).copy(),
                               "policy/proprio_normalized": U.normalize_proprio(np.asarray(obs["state"]).copy(), norm)})
                images = U.prepare_images_for_vla([obs["full_image"], obs["wrist_image"]], cfg)
                prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
                batches = [processor(prompt, image).to("cpu", dtype=torch.bfloat16) for image in images]
                arrays["model/pixel_values_bf16_as_float32"] = torch.cat([v["pixel_values"] for v in batches], dim=1).float().numpy()
                for name in ("input_ids", "attention_mask"):
                    if name in batches[0]:
                        arrays[f"model/{name}"] = batches[0][name].numpy()
                active["queries"][label] = {name: describe(value) for name, value in arrays.items()}
                active["query_steps"][label] = environment.steps
                return np.tile(np.asarray(FIXED_RAW_ACTION), (8, 1))

            def episode(*a, **kw):
                if state["manifest"] is None:
                    raise RuntimeError("Missing evaluator episode manifest")
                manifest = state["manifest"]
                episode_key = f"{manifest['task_id']}:{manifest['init_state_index']}"
                if episode_key in episodes:
                    raise RuntimeError("Duplicate probe episode")
                active = {"initial_state_sha256": manifest["init_state_sha256"], "queries": {}, "query_steps": {}}
                state.update(active=active, raw=None, env=a[1] if len(a) > 1 else kw["env"])
                try:
                    result = original_episode(*a, **kw)
                    if result[0] or active["query_steps"] != {"query0": 10, "after_fixed_8": 18}:
                        raise RuntimeError("Probe did not capture the required two real query boundaries")
                    episodes[episode_key] = active
                    return result
                finally:
                    state.update(active=None, raw=None, env=None)

            steps = dict(evaluator.TASK_MAX_STEPS)
            steps[evaluator.TaskSuite.LIBERO_SPATIAL] = 16
            with patched_attributes(evaluator, {"run_episode": episode, "prepare_observation": prepare,
                    "get_libero_env": new_env, "get_action": fixed_action, "log_message": log,
                    "save_rollout_video": lambda *a, **kw: None, "TASK_MAX_STEPS": steps}):
                for task_id in range(args.tasks):
                    if time.monotonic() - started > args.timeout_seconds:
                        raise TimeoutError("CPU bridge budget expired at a safe task boundary")
                    evaluator.run_task(cfg, suite, task_id, model=None, resize_size=evaluator.get_image_resize_size(cfg), processor=processor)
            if len(episodes) != args.tasks * args.resets:
                raise RuntimeError("Incomplete actual episode coverage")
            runs.append({"protocol": protocol, "repeat": repeat, "episodes": episodes})
    within, cross = compare_runs(runs)
    return {"gate": "PASS_CPU_PROTOCOL_OBSERVATION_BRIDGE" if all(v["repeat_equal"] for v in within.values()) else "FAIL_REPEATABILITY",
            "plan": specification, "sources": sources, "normalization_key": key,
            "within_protocol": within, "cross_protocol": cross, "runs": runs,
            "elapsed_seconds": time.monotonic() - started,
            "interpretation": "Only fixed-action observation and preprocessing differences are measured; GPU BF16/A0 bridge is still required."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--oft-root", type=Path)
    p.add_argument("--libero-root", type=Path)
    p.add_argument("--historical-evaluator", type=Path, default=HISTORICAL)
    p.add_argument("--historical-sha", default=HISTORICAL_SHA)
    p.add_argument("--model-seed", type=int, default=7)
    p.add_argument("--env-seed", type=int, default=0)
    p.add_argument("--tasks", type=int, default=10)
    p.add_argument("--resets", type=int, default=2)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--unnorm-key")
    p.add_argument("--timeout-seconds", type=int, default=900)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Never overwrite a probe receipt")
    if args.timeout_seconds < 1 or (not args.dry_run and (args.model_dir is None or not args.model_dir.is_dir())):
        p.error("Actual CPU probe requires a local processor directory and positive time budget")
    specification = plan(args)
    try:
        result = {"gate": "DRY_RUN_ONLY", "plan": specification} if args.dry_run else run(args, specification)
    except Exception as exc:
        result = {"gate": "FAIL_CPU_BRIDGE", "plan": specification,
                  "error_type": type(exc).__name__, "error": str(exc), "gpu_admission": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"gate": result["gate"], "output": str(args.output)}, ensure_ascii=False))
    if result["gate"].startswith("FAIL"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
