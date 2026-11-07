"""Audit the preregistered LIBERO cases without loading a policy or GPU.

The output contains task identities and hashes only. It never copies raw
initial states into the public Git repository or reads evaluation outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
INDICES = (5, 15, 25, 35, 45)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def source_commit(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def audit(source: Path) -> dict:
    # LIBERO's repository has an outer ``libero/`` source directory, while its
    # editable package metadata does not expose it on sys.path. Resolve the
    # caller-pinned source explicitly instead of relying on ambient PYTHONPATH.
    sys.path.insert(0, str(source))
    from libero.libero import benchmark, get_libero_path

    suites = benchmark.get_benchmark_dict()
    rows = []
    for name in SUITES:
        suite = suites[name]()
        if suite.n_tasks != 10:
            raise ValueError(f"Unexpected task count in {name}: {suite.n_tasks}")
        for task_id in range(10):
            task = suite.get_task(task_id)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            if not bddl.is_file():
                raise FileNotFoundError(bddl)
            states = suite.get_task_init_states(task_id)
            if len(states) != 50:
                raise ValueError(f"Expected 50 official states for {name}/{task_id}")
            selected = []
            for index in INDICES:
                state = np.asarray(states[index])
                if state.dtype.hasobject or state.size == 0 or not np.isfinite(state).all():
                    raise ValueError(f"Invalid initial state in {name}/{task_id}/{index}")
                selected.append({
                    "initial_state_index": index,
                    "state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                    "dtype": str(state.dtype),
                    "shape": list(state.shape),
                })
            if len({x["state_sha256"] for x in selected}) != len(INDICES):
                raise ValueError(f"Duplicate selected initial states in {name}/{task_id}")
            rows.append({
                "suite": name,
                "task_id": task_id,
                "task_description": task.language,
                "bddl_file": f"{task.problem_folder}/{task.bddl_file}",
                "bddl_sha256": sha_file(bddl),
                "official_initial_state_count": len(states),
                "selected": selected,
            })
    return {
        "gate": "PASS_LIBERO_FOUR_SUITE_FORTY_TASK_FIVE_STATE_IDENTITY_CPU",
        "libero_source_commit": source_commit(source),
        "suites": list(SUITES),
        "tasks": len(rows),
        "selected_cases": len(rows) * len(INDICES),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    report = audit(args.libero_source.resolve(strict=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("gate", "libero_source_commit", "tasks", "selected_cases")}))


if __name__ == "__main__":
    main()
