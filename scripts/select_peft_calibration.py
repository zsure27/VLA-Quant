"""Select calibration frames whose full trajectories belong to the PEFT train split."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-split", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()
    split = json.loads(args.trajectory_split.read_text())
    roles = {row["dataset_order_index"]: row["role"] for row in split["episodes"]}
    source_manifest = json.loads((args.source / "manifest.json").read_text())
    selected = []
    seen = set()
    for row in source_manifest["samples"]:
        if row["suite"] != "libero_spatial_no_noops" or row["episode"] in seen:
            continue
        if roles.get(row["episode"]) != "peft_train":
            continue
        selected.append(row)
        seen.add(row["episode"])
        if len(selected) == args.count:
            break
    if len(selected) != args.count:
        raise RuntimeError(f"only {len(selected)} eligible unique PEFT-train trajectories")
    args.output.mkdir(parents=True, exist_ok=False)
    output_rows = []
    for index, row in enumerate(selected):
        source = args.source / row["file"]
        target = args.output / f"sample-{index:04d}.npz"
        shutil.copy2(source, target)
        output_rows.append({**row, "file": target.name, "source_file": source.name,
                            "sha256": sha(target), "trajectory_role": "peft_train"})
    manifest = {"schema_version": "1.0", "selection": "unique Spatial trajectories restricted to P1 peft_train role",
                "trajectory_split": str(args.trajectory_split), "source": str(args.source), "samples": output_rows}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "samples": len(output_rows),
                      "episodes": [row["episode"] for row in output_rows]}, sort_keys=True))


if __name__ == "__main__":
    main()
