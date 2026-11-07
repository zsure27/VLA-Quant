"""Recover a GR00T video from immutable raw query images after codec failure."""
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


def run(source: Path, output: Path) -> None:
    source = source.resolve(strict=True)
    output = output.resolve()
    if output.exists() or source == output or source in output.parents:
        raise ValueError("Recovery output must be a new independent directory")
    original = source / "result.json"
    result = json.loads(original.read_text())
    if (result.get("gate") != "FAIL_GROOT_NATIVE_BF16_EPISODE"
            or result.get("error") != "needs one of codec_name or template"
            or result.get("error_type") != "ValueError"
            or result.get("condition") != "bf16" or result.get("success") is not True):
        raise ValueError("Original run is not the exact video codec failure")
    count = result.get("policy_queries")
    if not isinstance(count, int) or count < 1 or count != result.get("control_steps"):
        raise ValueError("Incomplete raw query/control trace")
    files = sorted(source.glob("query-*.npz"))
    if [p.name for p in files] != [f"query-{i:05d}.npz" for i in range(count)]:
        raise ValueError("Missing or duplicate query file")
    frames = []
    hashes = {}
    previous_post = None
    for index, file in enumerate(files):
        hashes[file.name] = sha(file)
        with np.load(file, allow_pickle=False) as row:
            frame = row["video.image"]
            action = row["action"]
            pre = row["robot0_eef_pos"]
            post = row["post_robot0_eef_pos"]
            done = bool(row["done"])
            if (frame.shape != (1, 256, 256, 3) or frame.dtype != np.uint8
                    or action.shape != (7,) or not np.isfinite(action).all()
                    or pre.shape != (3,) or post.shape != (3,)
                    or not np.isfinite(pre).all() or not np.isfinite(post).all()
                    or done != (index == count - 1)):
                raise ValueError(f"Corrupt raw query {index}")
            if previous_post is not None and not np.array_equal(previous_post, pre):
                raise ValueError(f"Robot position discontinuity before query {index}")
            previous_post = post.copy()
            frames.append(frame[0].copy())
    import imageio.v2 as imageio
    output.mkdir(parents=True)
    video = output / "rollout.mp4"
    imageio.mimsave(video, frames, fps=20, codec="libx264")
    report = {
        "gate": "PASS_GROOT_NATIVE_BF16_EPISODE_VIDEO_RECOVERED",
        "source_round_id": result["round_id"],
        "source_result_sha256": sha(original),
        "source_raw_query_files": hashes,
        "control_steps": count,
        "success": True,
        "video_sha256": sha(video),
        "video_bytes": video.stat().st_size,
        "source_run_preserved": True,
        "new_policy_queries": 0,
    }
    (output / "recovery.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "source_raw_query_files"}), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    run(a.source, a.output)


if __name__ == "__main__":
    main()
