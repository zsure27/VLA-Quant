"""Audit paired P2.5 Spatial500 outcomes directly from the raw archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows, shard_names = [], []
    with tarfile.open(args.archive, "r:gz") as archive:
        for member in archive.getmembers():
            if member.isfile() and member.name.endswith("/paired-episodes.jsonl"):
                stream = archive.extractfile(member)
                assert stream is not None
                rows.extend(json.loads(line) for line in stream if line.strip())
                shard_names.append(member.name)
    if len(shard_names) != 5 or len(rows) != 500:
        raise RuntimeError(f"Expected 5 shards and 500 pairs: {len(shard_names)}, {len(rows)}")
    identities = {(row["task_id"], row["init_state_index"]) for row in rows}
    if identities != {(task, state) for task in range(10) for state in range(50)}:
        raise RuntimeError("Paired episode identity coverage mismatch")
    successes = {case: sum(bool(row[case]) for row in rows) for case in ("C0", "C1", "C2")}
    if successes != summary["successes"]:
        raise RuntimeError("Raw outcomes disagree with series summary")

    def matrix(left: str, right: str) -> dict[str, int]:
        return {
            "both_success": sum(bool(row[left]) and bool(row[right]) for row in rows),
            f"{left}_only": sum(bool(row[left]) and not bool(row[right]) for row in rows),
            f"{right}_only": sum(not bool(row[left]) and bool(row[right]) for row in rows),
            "both_fail": sum(not bool(row[left]) and not bool(row[right]) for row in rows),
        }

    per_task = []
    for task in range(10):
        subset = [row for row in rows if row["task_id"] == task]
        per_task.append({"task_id": task, "episodes": len(subset),
                         "successes": {case: sum(bool(row[case]) for row in subset)
                                       for case in ("C0", "C1", "C2")}})
    headroom = [row for row in rows if not row["C0"] and row["C2"]]
    result = {
        "schema_version": "1.0", "classification": "historical/development paired; not offline_final_holdout",
        "archive": str(args.archive), "archive_sha256": sha256(args.archive),
        "summary_sha256": sha256(args.summary), "shard_episode_files": sorted(shard_names),
        "episodes_per_config": 500, "successes": successes,
        "paired_C0_C1": matrix("C0", "C1"), "paired_C0_C2": matrix("C0", "C2"),
        "paired_C1_C2": matrix("C1", "C2"),
        "headroom_C0_fail_C2_success": len(headroom),
        "headroom_rescued_by_C1": sum(bool(row["C1"]) for row in headroom),
        "headroom_still_failed_C1": sum(not bool(row["C1"]) for row in headroom),
        "headroom_resets": [{"task_id": row["task_id"], "init_state_index": row["init_state_index"],
                             "C1_success": bool(row["C1"])} for row in headroom],
        "per_task": per_task,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("successes", "paired_C0_C2",
                                               "headroom_C0_fail_C2_success", "headroom_rescued_by_C1")}))


if __name__ == "__main__":
    main()
