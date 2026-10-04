"""Content and trajectory checks for Recovery-LoRA inputs (no GPU required)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from qvla.on_policy_capture import observation_hash


def sample_fingerprint(path: Path) -> str:
    with np.load(path, allow_pickle=False) as data:
        return observation_hash(data["image"], data["wrist_image"], data["state"],
                                str(data["instruction"].item()))


def check_disjoint(left: list[Path], right: list[Path]) -> None:
    # Compare raw observation fields, ignoring filename/container/placeholder action.
    # Real evaluator smoke gates must additionally compare processed model inputs.
    overlap = {sample_fingerprint(p) for p in left} & {sample_fingerprint(p) for p in right}
    if overlap:
        raise ValueError(f"Policy-visible observation overlap: {len(overlap)}")
    def offline_sources(paths):
        identities = set()
        by_parent = {}
        for path in paths:
            if path.parent not in by_parent:
                manifest_path = path.parent / "manifest.json"
                payload = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
                by_parent[path.parent] = {r.get("file", r.get("name", r.get("sample"))): r for r in payload.get("samples", [])}
            row = by_parent[path.parent].get(path.name, {})
            # Simulator init_state_index and demonstration episode are distinct namespaces.
            if "init_state_index" not in row:
                index = row.get("dataset_order_index", row.get("episode"))
                if index is not None:
                    suite = row.get("suite", "libero_spatial").replace("_no_noops", "")
                    identities.add((suite, index))
        return identities
    shared_trajectories = offline_sources(left) & offline_sources(right)
    if shared_trajectories:
        raise ValueError(f"Offline trajectory overlap across recovery and probe: {sorted(shared_trajectories)}")


def audit_training_inputs(paths: list[Path], split_path: Path | None) -> dict:
    if not paths:
        raise ValueError("Empty recovery input set")
    parents = {p.parent.resolve() for p in paths}
    if len(parents) != 1:
        raise ValueError("Inputs must have one role manifest")
    manifest = json.loads((paths[0].parent / "manifest.json").read_text(encoding="utf-8"))
    rows = manifest.get("samples", [])
    by_file = {r.get("file", r.get("sample", r.get("name"))): r for r in rows}
    if len(by_file) != len(rows):
        raise ValueError("Duplicate manifest filenames")
    split = None
    if split_path:
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        split = {r["dataset_order_index"]: r for r in payload["episodes"]}
    fingerprints, reset_keys = [], set()
    for path in paths:
        row = by_file.get(path.name)
        if row is None:
            raise ValueError(f"Missing role record: {path.name}")
        expected_hash = row.get("sample_sha256", row.get("sha256"))
        if not expected_hash or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Input SHA mismatch: {path.name}")
        role = row.get("trajectory_role", row.get("role", manifest.get("role")))
        if role == "student_state_train":
            reset_keys.add((row["task_id"], row["init_state_index"]))
        elif role == "peft_train":
            if split is None:
                raise ValueError("Offline recovery inputs require --trajectory-split")
            index = row.get("dataset_order_index", row.get("episode"))
            original = split.get(index)
            if not original or original["role"] != "peft_train" or original["instruction"] != row["instruction"]:
                raise ValueError(f"Offline trajectory role mismatch: {path.name}")
            if row.get("trajectory_id") and row["trajectory_id"] != original["trajectory_id"]:
                raise ValueError("Trajectory identity mismatch")
        else:
            raise ValueError(f"Recovery input role not permitted: {role}")
        fingerprint = sample_fingerprint(path)
        if row.get("observation_sha256") and row["observation_sha256"] != fingerprint:
            raise ValueError("Observation SHA mismatch")
        fingerprints.append(fingerprint)
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("Duplicate observations inflate recovery sample count")
    return {"count": len(paths), "unique_observations": len(fingerprints),
            "student_training_resets": sorted(reset_keys), "manifest": str(paths[0].parent / "manifest.json")}
