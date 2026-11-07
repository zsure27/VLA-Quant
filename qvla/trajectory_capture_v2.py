"""Independent observer for OFT run_episode; never edits a frozen evaluator.

Install after backend selection. Uses existing observations, never extra env.step,
get_observation or model forward calls. GPU equivalence admission remains required.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import time
import numpy as np

from qvla.on_policy_capture import observation_hash
from qvla.trajectory_diagnostics_v1 import finite, quat_unit


class CaptureFailure(BaseException):
    """Bypass evaluator's broad Exception catch; never turn an audit error into SR."""


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


@contextmanager
def observe_episode(module, root, identity):
    """Context around evaluation. EPISODE_MANIFEST identifies actual task/reset."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    original_episode, original_action, original_log = module.run_episode, module.get_action, module.log_message
    manifest, active = {}, {}

    def log(message, *args, **kwargs):
        if str(message).startswith("EPISODE_MANIFEST "):
            manifest.clear()
            manifest.update(json.loads(str(message).split(" ", 1)[1]))
        return original_log(message, *args, **kwargs)

    def action(cfg, model, observation, instruction, **kwargs):
        if not active:
            raise CaptureFailure("Query outside traced episode")
        # Existing get_action can normalize state in place; capture BEFORE calling it.
        image = np.asarray(observation["full_image"]).copy()
        wrist = np.asarray(observation["wrist_image"]).copy()
        state = np.asarray(observation["state"]).copy()
        input_hash = observation_hash(image, wrist, state, instruction)
        t = time.perf_counter()
        output = original_action(cfg, model, observation, instruction, **kwargs)
        elapsed = time.perf_counter()-t  # Wall time, NOT CUDA kernel latency.
        try:
            chunk = finite(output)
            if chunk.ndim != 2 or chunk.shape[1] != 7 or len(chunk) != active["chunk_size"]:
                raise ValueError("Unsupported action chunk shape/execution prefix")
            q = active["query"]+1
            np.savez_compressed(active["folder"]/f"query-{q:05d}.npz", image=image,
                                wrist_image=wrist, state=state, instruction=np.asarray(instruction),
                                raw_policy_chunk=np.asarray(output).copy())
            active.update(query=q, slot=0)
            active["emit"]({"kind": "policy_query", "query_index": q,
                            "next_control_step": active["steps"], "input_sha256": input_hash,
                            "raw_state_dtype": str(state.dtype), "policy_call_wall_s": elapsed})
        except Exception as exc:
            raise CaptureFailure(str(exc)) from exc
        return output

    def episode(*args, **kwargs):
        bound = inspect.signature(original_episode).bind(*args, **kwargs)
        bound.apply_defaults()
        cfg, env = bound.arguments["cfg"], bound.arguments["env"]
        required = ("task_id", "init_state_index", "model_seed", "env_seed", "init_state_sha256")
        if any(k not in manifest for k in required):
            raise CaptureFailure("Actual paired EPISODE_MANIFEST required")
        if cfg.seed_protocol != "paired" or cfg.initial_states_path != "DEFAULT":
            raise CaptureFailure("Only audited paired default-reset evaluator supported")
        meta = dict(identity, suite=cfg.task_suite_name, task_id=manifest["task_id"],
                    reset_index=manifest["init_state_index"], chunk_size=cfg.num_open_loop_steps,
                    status="running", **{k: manifest[k] for k in required[2:]})
        manifest.clear()  # A later episode must provide its own actual manifest.
        folder = root/f"task-{meta['task_id']:02d}-reset-{meta['reset_index']:03d}"
        folder.mkdir(exist_ok=False)
        write_json(folder/"episode.json", meta)
        stream = (folder/"steps.jsonl").open("x", encoding="utf-8")
        step, set_state = env.step, env.set_init_state
        get_obs = getattr(env, "get_observation", None)
        if get_obs is None and bound.arguments.get("initial_state") is None:
            raise CaptureFailure("Evaluator would need unavailable get_observation")

        def emit(row):
            stream.write(json.dumps(row, allow_nan=False)+"\n")
            stream.flush()

        active.update(folder=folder, steps=0, query=-1, slot=0, chunk_size=cfg.num_open_loop_steps,
                      calls=0, obs=None, emit=emit)

        def capture_obs(fn, *a, **kw):
            out = fn(*a, **kw)
            active["obs"] = out
            return out

        def observed_step(command):
            before = active["obs"]
            out = step(command)
            after = out[0]
            active["obs"] = after
            active["calls"] += 1
            if active["calls"] <= cfg.num_steps_wait:
                emit({"kind": "settle_step", "settle_step": active["calls"]-1})
                return out
            try:
                if before is None or active["query"] < 0 or active["slot"] >= active["chunk_size"]:
                    raise ValueError("Missing pre-state/query or invalid queue position")
                emit({"kind": "control_step", "control_step": active["steps"],
                      "query_index": active["query"], "chunk_position": active["slot"],
                      "pre_position_m": finite(before["robot0_eef_pos"], (3,)).tolist(),
                      "post_position_m": finite(after["robot0_eef_pos"], (3,)).tolist(),
                      "post_quaternion_xyzw": quat_unit(after["robot0_eef_quat"]).tolist(),
                      "gripper_qpos": finite(after["robot0_gripper_qpos"]).tolist(),
                      "controller_command": finite(command, (7,)).tolist(), "done": bool(out[2])})
                active["steps"] += 1
                active["slot"] += 1
            except Exception as exc:
                raise CaptureFailure(str(exc)) from exc
            return out

        env.step = observed_step
        env.set_init_state = lambda *a, **kw: capture_obs(set_state, *a, **kw)
        if get_obs is not None:
            env.get_observation = lambda *a, **kw: capture_obs(get_obs, *a, **kw)
        try:
            result = original_episode(*args, **kwargs)
            # The old evaluator catches Exception; a false return before horizon is NOT timeout.
            if result[0]:
                meta["status"] = "success"
            elif active["steps"] == module.TASK_MAX_STEPS[cfg.task_suite_name]:
                meta["status"] = "timeout"
            else:
                raise CaptureFailure("Evaluator returned early without success; possible swallowed exception")
            return result
        except BaseException as exc:
            meta.update(status="interrupted_or_error", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            env.step, env.set_init_state = step, set_state
            if get_obs is not None:
                env.get_observation = get_obs
            meta.update(control_steps=active["steps"], policy_queries=active["query"]+1)
            write_json(folder/"episode.json", meta)
            stream.close()
            active.clear()

    module.run_episode, module.get_action, module.log_message = episode, action, log
    try:
        yield
    finally:
        module.run_episode, module.get_action, module.log_message = original_episode, original_action, original_log
