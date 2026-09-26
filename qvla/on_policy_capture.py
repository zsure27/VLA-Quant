"""Crash-tolerant capture helpers for P2.5 same-observation diagnostics."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def action_metrics(student: np.ndarray, teacher: np.ndarray) -> dict[str, Any]:
    """Split same-observation 8x7 disagreement by executed action component."""
    if student.shape != teacher.shape or student.ndim != 2 or student.shape[1] != 7:
        raise RuntimeError(f"Action shape mismatch: {student.shape} vs {teacher.shape}")
    delta = student.astype(np.float64) - teacher.astype(np.float64)
    student_gripper = np.sign(2 * student[:, 6] - 1)
    teacher_gripper = np.sign(2 * teacher[:, 6] - 1)
    return {
        "chunk_mse": float(np.mean(delta**2)),
        "chunk_rmse": float(np.sqrt(np.mean(delta**2))),
        "position_rmse": float(np.sqrt(np.mean(delta[:, 0:3] ** 2))),
        "rotation_rmse": float(np.sqrt(np.mean(delta[:, 3:6] ** 2))),
        "gripper_continuous_rmse": float(np.sqrt(np.mean(delta[:, 6] ** 2))),
        "gripper_disagreement": float(np.mean(student_gripper != teacher_gripper)),
        "student_gripper_signed_margin": (2 * student[:, 6] - 1).tolist(),
        "teacher_gripper_signed_margin": (2 * teacher[:, 6] - 1).tolist(),
        "rmse_per_step": np.sqrt(np.mean(delta**2, axis=1)).tolist(),
        "rmse_per_dimension": np.sqrt(np.mean(delta**2, axis=0)).tolist(),
    }


def _array_hash(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def observation_hash(image: np.ndarray, wrist_image: np.ndarray, state: np.ndarray, instruction: str) -> str:
    digest = hashlib.sha256()
    for value in (image, wrist_image, state):
        digest.update(_array_hash(value).encode())
    digest.update(instruction.encode("utf-8"))
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class H17SummaryRecorder:
    """Keep causal pre-block18 pooled features for the current policy query."""

    def __init__(self, model: Any):
        self.value: dict[str, np.ndarray] | None = None
        layer = model.language_model.model.layers[17]
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        value = tensor.detach().float().cpu()
        if value.ndim != 3 or value.shape[0] != 1:
            raise RuntimeError(f"Unexpected H17 shape: {tuple(value.shape)}")
        self.value = {
            "h17_token_mean": value.mean(dim=1).squeeze(0).numpy(),
            "h17_last_token": value[0, -1].numpy(),
        }

    def reset(self) -> None:
        self.value = None

    def take(self) -> dict[str, np.ndarray]:
        if self.value is None:
            raise RuntimeError("H17 did not execute during policy query")
        return self.value

    def close(self) -> None:
        self.handle.remove()


class OnPolicyObservationCapture:
    """Write every student-query observation and an append-only episode event stream."""

    def __init__(self, log_dir: Path, chunk_execution_steps: int):
        self.root = log_dir / "policy-observations"
        self.root.mkdir(parents=True, exist_ok=False)
        self.events = (log_dir / "on-policy-events.jsonl").open("x", encoding="utf-8")
        self.chunk_execution_steps = int(chunk_execution_steps)
        self.episode_serial = -1
        self.query_in_episode = 0
        self.active = False
        self.total_queries = 0

    def _event(self, value: dict[str, Any]) -> None:
        self.events.write(json.dumps(value, sort_keys=True) + "\n")
        self.events.flush()
        os.fsync(self.events.fileno())

    def begin_episode(self, task: str) -> None:
        if self.active:
            raise RuntimeError("Previous capture episode is still active")
        self.episode_serial += 1
        self.query_in_episode = 0
        self.active = True
        self._event({"record_type": "episode_start", "episode_serial": self.episode_serial, "task": task})

    def record_query(
        self,
        observation: dict[str, Any],
        task: str,
        student_action: np.ndarray,
        h17: dict[str, np.ndarray],
    ) -> None:
        if not self.active:
            raise RuntimeError("Policy query occurred outside a captured episode")
        image = np.asarray(observation["full_image"])
        wrist_image = np.asarray(observation["wrist_image"])
        state = np.asarray(observation["state"], dtype=np.float32)
        action = np.asarray(student_action, dtype=np.float32)
        if action.ndim != 2 or action.shape[1] != 7 or not np.isfinite(action).all():
            raise RuntimeError(f"Invalid student action chunk: {action.shape}")
        name = f"query-{self.total_queries:06d}.npz"
        destination = self.root / name
        with tempfile.NamedTemporaryFile(dir=self.root, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            np.savez_compressed(
                temporary,
                image=image,
                wrist_image=wrist_image,
                state=state,
                instruction=np.asarray(task),
                student_action=action,
                **{key: np.asarray(value, dtype=np.float32) for key, value in h17.items()},
            )
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        self._event({
            "record_type": "query",
            "episode_serial": self.episode_serial,
            "query_in_episode": self.query_in_episode,
            "executed_step_start": self.query_in_episode * self.chunk_execution_steps,
            "chunk_execution_steps": self.chunk_execution_steps,
            "task": task,
            "file": f"policy-observations/{name}",
            "file_sha256": file_sha256(destination),
            "observation_sha256": observation_hash(image, wrist_image, state, task),
        })
        self.query_in_episode += 1
        self.total_queries += 1

    def end_episode(self, success: bool, aborted: bool = False) -> None:
        if not self.active:
            return
        self._event({
            "record_type": "episode_end",
            "episode_serial": self.episode_serial,
            "queries": self.query_in_episode,
            "success": bool(success),
            "aborted": bool(aborted),
        })
        self.active = False

    def close(self) -> None:
        if self.active:
            self.end_episode(False, aborted=True)
        self.events.close()
