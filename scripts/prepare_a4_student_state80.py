"""Extract exactly query0/1 from 40 completed, freshly recorded A4 training episodes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qvla.extended_peft import sha, canonical_sha, student_contract
from qvla.on_policy_capture import observation_hash


def prepare(run, output):
    invocation = json.loads((run / "invocation.json").read_text())
    command = invocation["command"]
    def flag(name): return command[command.index(name) + 1]
    for key, value in (("--model-id", "A4"), ("--initial-state-offset", "0"),
                       ("--num_trials_per_task", "4"), ("--seed-protocol", "paired")):
        if flag(key) != value: raise ValueError("A4 capture invocation differs from the preregistration")
    if "--trace-observations" not in command or "--trace-actions" not in command:
        raise ValueError("Capture needs both observation and action traces")
    if (run / "exit-code.txt").read_text().strip() != "0": raise ValueError("Capture did not complete")
    logs = list(run.glob("EVAL-*.txt"))
    if len(logs) != 1: raise ValueError("Require exactly one evaluation log")
    pairs = [json.loads(m.group(1)) for m in re.finditer(r"EPISODE_MANIFEST (\{.*\})", logs[0].read_text())]
    if len(pairs) != 40 or {(r["task_id"], r["init_state_index"]) for r in pairs} != {(t, r) for t in range(10) for r in range(4)}:
        raise ValueError("Capture must cover all 40 training task/reset pairs once")
    events = [json.loads(line) for line in (run / "on-policy-events.jsonl").read_text().splitlines()]
    ends = [r for r in events if r["record_type"] == "episode_end"]
    if len(ends) != 40 or {r["episode_serial"] for r in ends} != set(range(40)) or any(r["aborted"] or r["queries"] < 2 for r in ends):
        raise ValueError("Capture has aborted/short/missing episodes; cannot substitute A3 data")
    queries = sorted((r for r in events if r["record_type"] == "query" and r["query_in_episode"] in (0, 1)),
                     key=lambda r: (r["episode_serial"], r["query_in_episode"]))
    if len(queries) != 80: raise ValueError("Expected 80 captured observations")
    if output.exists(): raise ValueError("Never overwrite training data")
    output.mkdir(parents=True)
    rows = []
    for index, row in enumerate(queries):
        source = run / row["file"]
        if not source.resolve().is_relative_to(run.resolve()) or sha(source) != row["file_sha256"]:
            raise ValueError("Captured source path/hash mismatch")
        with np.load(source, allow_pickle=False) as sample:
            image, wrist = sample["image"], sample["wrist_image"]
            state, action = sample["state"].astype(np.float32), sample["student_action"].astype(np.float32)
            instruction = str(sample["instruction"].item())
        if action.shape != (8, 7) or not np.isfinite(action).all() or row["chunk_execution_steps"] != 8:
            raise ValueError("A4 student must produce finite real 8x7 chunks")
        if observation_hash(image, wrist, state, instruction) != row["observation_sha256"]:
            raise ValueError("Raw observation hash mismatch")
        target = output / f"sample-{index:05d}.npz"
        np.savez_compressed(target, image=image, wrist_image=wrist, state=state,
                            instruction=np.array(instruction), action=action[0])
        pair = pairs[row["episode_serial"]]
        rows.append({"file": target.name, "task_id": pair["task_id"], "init_state_index": pair["init_state_index"],
            "episode_serial": row["episode_serial"], "query_in_episode": row["query_in_episode"],
            "init_state_sha256": pair["init_state_sha256"], "sample_sha256": sha(target),
            "observation_sha256": row["observation_sha256"], "source": row["file"], "source_sha256": sha(source)})
    manifest = {"schema_version": "2.0", "role": "student_state_train", "source_model_id": "A4",
        "source_run_sha256": canonical_sha({name: sha(run / name) for name in
            ("invocation.json", "on-policy-events.jsonl", "policy-queries.jsonl", logs[0].name)}),
        "count": 80, "samples": rows, "dataset_action_role": "placeholder; BF16 same-observation teacher queried during smoke",
        "selection": "10 tasks x reset0-3 x query0-1; no closed-loop success filtering"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    student_contract(output, "B3")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args(); prepare(a.run, a.output)
