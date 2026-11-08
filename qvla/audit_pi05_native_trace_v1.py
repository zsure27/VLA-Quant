"""Independently audit a saved native pi0.5 LIBERO trajectory on CPU."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def audit(root: Path, plan_path: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    result = json.loads((root / "result.json").read_text())
    if result.get("gate") != "PASS_PI05_NATIVE_BF16_EPISODE":
        raise ValueError("Original run did not pass")
    for key in ("round_id", "suite", "task_id", "reset_index", "model_seed", "condition"):
        if result.get(key) != plan.get(key):
            raise ValueError(f"Plan mismatch: {key}")
    if root.resolve() != Path(plan["output"]).resolve() or result["plan_sha256"] != sha(plan_path):
        raise ValueError("Plan path or content mismatch")
    step_paths = sorted(root.glob("step-*.npz"))
    query_paths = sorted(root.glob("query-*.npz"))
    if (len(step_paths) != result["control_steps"] or len(query_paths) != result["policy_queries"]
            or len(step_paths) < 1 or len(query_paths) < 1):
        raise ValueError("Incomplete raw trace")
    if [p.name for p in step_paths] != [f"step-{i:05d}.npz" for i in range(len(step_paths))]:
        raise ValueError("Missing or duplicate control step")
    if [p.name for p in query_paths] != [f"query-{i:05d}.npz" for i in range(len(query_paths))]:
        raise ValueError("Missing or duplicate policy query")
    if {p.name: sha(p) for p in step_paths} != result["step_files"]:
        raise ValueError("Step file digest mismatch")
    if {p.name: sha(p) for p in query_paths} != result["query_files"]:
        raise ValueError("Query file digest mismatch")
    if sha(root / "rollout.mp4") != result["video_sha256"]:
        raise ValueError("Video digest mismatch")
    pose = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
    queries = [read(p) for p in query_paths]
    starts = []
    for index, query in enumerate(queries):
        start = int(query["control_step"])
        if start != index * 5 or query["image"].shape != (224, 224, 3) or query["wrist_image"].shape != (224, 224, 3):
            raise ValueError("Unexpected native query cadence or image")
        if query["state"].shape != (8,) or query["raw_policy_chunk"].shape[1:] != (7,):
            raise ValueError("Malformed state or action chunk")
        if query["raw_policy_chunk"].shape[0] < 5 or not np.isfinite(query["raw_policy_chunk"]).all():
            raise ValueError("Nonfinite or short action chunk")
        if query["prompt"].shape != () or not str(query["prompt"].item()):
            raise ValueError("Missing instruction")
        for name in pose:
            if not np.isfinite(query[name]).all():
                raise ValueError("Nonfinite query pose")
        starts.append(start)
    last = None
    done_count = 0
    for index, path in enumerate(step_paths):
        row = read(path)
        if row["action"].shape != (7,) or not np.isfinite(row["action"]).all():
            raise ValueError("Malformed executed action")
        query_index, slot = divmod(index, 5)
        if not np.array_equal(row["action"], queries[query_index]["raw_policy_chunk"][slot]):
            raise ValueError("Executed action differs from saved policy chunk")
        if slot == 0:
            for name in pose:
                if not np.array_equal(row[name], queries[query_index][name]):
                    raise ValueError("Query input pose differs from control step")
        if last is not None:
            for name in pose:
                if not np.array_equal(row[name], last[f"post_{name}"]):
                    raise ValueError("Broken consecutive pose trace")
        for name in pose:
            if not np.isfinite(row[name]).all() or not np.isfinite(row[f"post_{name}"]).all():
                raise ValueError("Nonfinite pose trace")
        done_count += bool(row["done"].item())
        if bool(row["done"].item()) and index != len(step_paths) - 1:
            raise ValueError("Steps continued after terminal signal")
        last = row
    if done_count != int(result["success"]):
        raise ValueError("Terminal signal and success differ")
    return {"gate": "PASS_PI05_NATIVE_BF16_TRACE_AUDIT", "round_id": plan["round_id"],
            "plan_sha256": sha(plan_path), "result_sha256": sha(root / "result.json"),
            "control_steps": len(step_paths), "policy_queries": len(query_paths),
            "success": result["success"], "video_sha256": result["video_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(args.root.resolve(strict=True), args.plan.resolve(strict=True))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
