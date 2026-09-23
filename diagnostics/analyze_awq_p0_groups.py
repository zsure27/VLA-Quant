"""Build the paired episode matrix for the 2026-09-23 P0 visual-group grid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_run(run: Path):
    log = next(run.glob("*/EVAL-*.txt"))
    rows, pending = [], None
    for line in log.read_text(encoding="utf-8").splitlines():
        if line.startswith("EPISODE_MANIFEST "):
            pending = json.loads(line.split(" ", 1)[1])
        elif line.startswith("Success: "):
            if pending is None:
                raise RuntimeError("success without manifest")
            rows.append({**pending, "success": line.endswith("True")})
            pending = None
    if len(rows) != 50:
        raise RuntimeError(f"{run.name}: expected 50 episodes, got {len(rows)}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--both-g64-run", type=Path, required=True)
    args = parser.parse_args()
    labels = {
        "dino-g64-siglip-g128": "dino64_siglip128",
        "dino-g128-siglip-g64": "dino128_siglip64",
        "visual-g128": "dino128_siglip128",
    }
    runs = {}
    for run in args.root.iterdir():
        if run.is_dir():
            for marker, label in labels.items():
                if marker in run.name:
                    runs[label] = read_run(run)
    runs["dino64_siglip64"] = read_run(args.both_g64_run)
    expected = set(labels.values()) | {"dino64_siglip64"}
    if set(runs) != expected:
        raise RuntimeError(f"missing runs: {expected - set(runs)}")
    order = list(runs)
    keys = [[(r["task_id"], r["init_state_index"], r["init_state_sha256"], r["env_seed"], r["model_seed"])
             for r in runs[label]] for label in order]
    if any(value != keys[0] for value in keys[1:]):
        raise RuntimeError("episode manifests are not strictly paired")
    episodes = []
    for index, key in enumerate(keys[0]):
        episodes.append({"task_id": key[0], "initial_state_index": key[1], "initial_state_sha256": key[2],
                         "success": {label: runs[label][index]["success"] for label in order}})
    summary = {
        "evidence_class": "measured_paired_closed_loop",
        "task_suite": "libero_spatial",
        "initial_state_range": [5, 9],
        "language_recipe": "W2 G64 attention-only no-clip; MLP clipping retained",
        "successes": {label: sum(row["success"] for row in runs[label]) for label in order},
        "episodes": episodes,
        "strict_manifest_pairing": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["successes"], sort_keys=True))


if __name__ == "__main__":
    main()
