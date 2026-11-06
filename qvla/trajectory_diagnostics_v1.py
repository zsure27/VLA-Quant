"""CPU-only geometry and descriptive alarms; no model or CUDA imports.

Pose differences between independent rollouts are NOT isolated quantization error.
Quaternions use LIBERO xyzw; relative rotation is expressed in world coordinates.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def finite(value, shape=None):
    a = np.asarray(value, dtype=np.float64)
    if (shape is not None and a.shape != shape) or not np.isfinite(a).all():
        raise ValueError("Malformed or nonfinite trajectory field")
    return a


def quat_unit(value):
    q = finite(value, (4,))
    n = np.linalg.norm(q)
    if abs(n - 1) > 1e-3:
        raise ValueError("Expected unit xyzw quaternion")
    return q / n


def quat_mul(a, b):
    av, bv = a[:3], b[:3]
    return np.r_[a[3]*bv + b[3]*av + np.cross(av, bv), a[3]*b[3]-np.dot(av, bv)]


def rotation_difference(student, reference):
    """Log(R_student R_reference.T), radians, principal branch, world frame."""
    qs, qr = quat_unit(student), quat_unit(reference)
    delta = quat_mul(qs, np.r_[-qr[:3], qr[3]])
    if delta[3] < 0:
        delta = -delta
    length = np.linalg.norm(delta[:3])
    if length < 1e-12:
        return np.zeros(3)
    return delta[:3] / length * (2*np.arctan2(length, max(0., delta[3])))


def sustained_windows(values, threshold, persistence=3):
    """Past/current-only threshold alarm. Onset is known only at confirmation."""
    x = finite(values)
    if x.ndim != 1 or threshold <= 0 or persistence < 1:
        raise ValueError("Invalid alarm settings")
    windows, start = [], None
    for i, v in enumerate(x):
        if v > threshold and start is None:
            start = i
        if (v <= threshold or i == len(x)-1) and start is not None:
            end = i-1 if v <= threshold else i
            if end-start+1 >= persistence:
                windows.append({"onset_index": start, "confirmed_index": start+persistence-1,
                                "end_index": end, "peak": float(np.max(x[start:end+1]))})
            start = None
    return windows


def rolling_spread(vectors, window=16):
    """Local temporal scatter, NOT variance across independent episodes."""
    a = finite(vectors)
    if a.ndim != 2 or a.shape[1] != 3 or window < 2:
        raise ValueError("Expected Nx3 vectors and window>=2")
    out = []
    for i in range(len(a)):
        v = a[max(0, i-window+1):i+1]
        out.append(float(np.sqrt(np.mean(np.sum((v-v.mean(axis=0))**2, axis=1)))))
    return out


def load_steps(path):
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x]
    rows = [r for r in rows if r["kind"] == "control_step"]
    indices = [r["control_step"] for r in rows]
    if indices != list(range(len(rows))):
        raise ValueError("Duplicate, missing or unordered control steps")
    return rows


def compare_episode(candidate_dir, reference_dir, thresholds):
    cd, rd = Path(candidate_dir), Path(reference_dir)
    c, r = [json.loads((p/"episode.json").read_text(encoding="utf-8")) for p in (cd, rd)]
    if any(m["status"] not in ("success", "timeout", "interrupted_or_error") for m in (c,r)):
        raise ValueError("Only terminal episode records can be analyzed")
    for key in ("suite", "task_id", "reset_index", "model_seed", "env_seed",
                "init_state_sha256", "checkpoint_sha256", "cohort_id", "chunk_size"):
        if c[key] != r[key]:
            raise ValueError("Unmatched episode identity: " + key)
    if c["model_id"] == r["model_id"] and cd.resolve() == rd.resolve():
        raise ValueError("Cannot compare an episode with itself")
    cs, rs = load_steps(cd/"steps.jsonl"), load_steps(rd/"steps.jsonl")
    n = min(len(cs), len(rs))
    if not n:
        raise ValueError("No common valid control steps")
    trans, rot = [], []
    for a, b in zip(cs[:n], rs[:n]):
        trans.append((finite(a["post_position_m"], (3,))-finite(b["post_position_m"], (3,))).tolist())
        rot.append(rotation_difference(a["post_quaternion_xyzw"], b["post_quaternion_xyzw"]).tolist())
    tn, rn = np.linalg.norm(trans, axis=1), np.linalg.norm(rot, axis=1)*180/np.pi
    result = {"schema_version": 1, "data_kind": c["data_kind"], "suite": c["suite"],
              "task_id": c["task_id"], "reset_index": c["reset_index"],
              "candidate": c["model_id"], "reference": r["model_id"],
              "candidate_status": c["status"], "reference_status": r["status"],
              "common_steps": n, "candidate_steps": len(cs), "reference_steps": len(rs),
              "censored": len(cs) != len(rs), "translation_vector_m": trans,
              "rotation_vector_rad_world": rot, "translation_norm_m": tn.tolist(),
              "rotation_norm_deg": rn.tolist(), "local_translation_spread_m": rolling_spread(trans),
              "translation_windows": sustained_windows(tn, thresholds["translation_m"], thresholds["persistence"]),
              "rotation_windows": sustained_windows(rn, thresholds["rotation_deg"], thresholds["persistence"]),
              "thresholds": thresholds,
              "interpretation": "Matched-reset closed-loop pose difference; association, not isolated quantization error or causal failure proof."}
    if c["data_kind"] != r["data_kind"]:
        raise ValueError("Cannot mix measured and synthetic episodes")
    return result


def same_observation_chunk(student, teacher, student_hash, teacher_hash):
    if not student_hash or student_hash != teacher_hash:
        raise ValueError("Teacher and student input hashes differ")
    a, b = finite(student), finite(teacher)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 7:
        raise ValueError("Expected matching Hx7 raw policy chunks")
    # Raw rotation COMMAND difference is not an SO(3) pose error.
    return {"translation_command_l2": np.linalg.norm(a[:, :3]-b[:, :3], axis=1).tolist(),
            "rotation_command_l2": np.linalg.norm(a[:, 3:6]-b[:, 3:6], axis=1).tolist(),
            "gripper_continuous_absolute": np.abs(a[:, 6]-b[:, 6]).tolist(),
            "note": "Same query input and same chunk position; controller processing is audited separately."}
