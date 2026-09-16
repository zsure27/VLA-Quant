"""Analyze a complete paired BF16/W4 gate without counting repeated states twice."""
import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def read_run(path):
    if (path / "exit-code.txt").read_text().strip() != "0":
        raise ValueError(f"Run did not exit successfully: {path}")
    logs = list(path.glob("EVAL-*.txt"))
    if len(logs) != 1:
        raise ValueError("Expected exactly one evaluation log")
    text = logs[0].read_text()
    if "Episode error:" in text:
        raise ValueError("Episode exception must not be counted as policy failure")
    manifests = [json.loads(x) for x in re.findall(r"^EPISODE_MANIFEST (.+)$", text, re.M)]
    outcomes = [x == "True" for x in re.findall(r"^Success: (True|False)$", text, re.M)]
    if len(manifests) != 50 or len(outcomes) != 50:
        raise ValueError("Expected 50 completed episodes")
    rows = {}
    for manifest, success in zip(manifests, outcomes):
        key = (manifest["task_id"], manifest["init_state_index"])
        if key in rows:
            raise ValueError("Duplicate episode key")
        rows[key] = {"manifest": manifest, "success": success}
    if set(rows) != {(task, state) for task in range(10) for state in range(5)}:
        raise ValueError("Unexpected task/state selection")
    return rows


def stats(bf16, w4, indices):
    b = bf16[:, indices]
    w = w4[:, indices]
    delta = w.astype(float) - b.astype(float)
    rng = np.random.default_rng(20260916)
    draws = rng.integers(0, len(indices), size=(10000, 10, len(indices)))
    sampled = delta[np.arange(10)[None, :, None], draws].mean(axis=(1, 2))
    low, high = np.quantile(sampled, [0.025, 0.975])
    bf16_only = int((b & ~w).sum())
    w4_only = int((~b & w).sum())
    discordant = bf16_only + w4_only
    p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(bf16_only, w4_only) + 1)) / 2**discordant)
    return {"episodes": int(b.size), "bf16_successes": int(b.sum()), "w4_successes": int(w.sum()),
            "bf16_success_rate": float(b.mean()), "w4_success_rate": float(w.mean()),
            "paired_difference_w4_minus_bf16": float(delta.mean()),
            "bf16_only_success": bf16_only, "w4_only_success": w4_only,
            "two_sided_exact_mcnemar_p": p,
            "conditional_stratified_bootstrap_95ci": [float(low), float(high)],
            "ci_note": "10 fixed tasks; resample observed initial states within each task. Not proof of equivalence or a benchmark-wide confidence guarantee."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    b = read_run(args.batch / "bf16")
    w = read_run(args.batch / "w4")
    if any(b[key]["manifest"] != w[key]["manifest"] for key in b):
        raise ValueError("Paired manifest mismatch")
    bf16 = np.array([[b[(t, i)]["success"] for i in range(5)] for t in range(10)])
    w4 = np.array([[w[(t, i)]["success"] for i in range(5)] for t in range(10)])
    result = {"batch": str(args.batch), "manifest_equality": True,
              "all_50": stats(bf16, w4, list(range(5))),
              "new_initial_states_1_to_4": stats(bf16, w4, list(range(1, 5))),
              "repeated_initial_state_0": {"bf16": bf16[:, 0].tolist(), "w4": w4[:, 0].tolist()},
              "per_task": [{"task_id": t, "bf16": int(bf16[t].sum()), "w4": int(w4[t].sum())} for t in range(10)],
              "unmeasured": ["GT action L1", "ACT hidden cosine", "peak VRAM", "synchronized warmed latency", "packed real quant parity"],
              "scope": "BF16-carried simulated AWQ W4A16; no recovery training. Five initial states per task; index 0 overlaps the prior ten-state screening."}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / "paired_episodes.csv").open("w", newline="") as stream:
        fields = ["task_id", "init_state_index", "bf16_success", "w4_success", "model_seed", "env_seed", "init_state_sha256", "previously_screened"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key in sorted(b):
            m = b[key]["manifest"]
            writer.writerow({"task_id": key[0], "init_state_index": key[1], "bf16_success": b[key]["success"],
                             "w4_success": w[key]["success"], "model_seed": m["model_seed"], "env_seed": m["env_seed"],
                             "init_state_sha256": m["init_state_sha256"], "previously_screened": key[1] == 0})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1, 1.3]})
    codes = bf16.astype(int) + 2 * w4.astype(int)
    axes[0].imshow(codes, vmin=0, vmax=3, cmap=ListedColormap(["#bd6666", "#d89439", "#5e91c9", "#46a77a"]), aspect="auto")
    axes[0].set(xticks=range(5), yticks=range(10), xlabel="Initial state (0 repeats prior screening)", ylabel="Task ID", title="Paired outcome: BF16 / W4")
    for t in range(10):
        for i in range(5):
            axes[0].text(i, t, f"{int(bf16[t,i])}/{int(w4[t,i])}", ha="center", va="center", color="white")
    y = np.arange(10)
    axes[1].barh(y - .18, bf16.sum(axis=1), height=.35, label="BF16")
    axes[1].barh(y + .18, w4.sum(axis=1), height=.35, label="AWQ W4A16")
    axes[1].set(yticks=y, xlabel="Successes / 5", ylabel="Task ID", xlim=(0, 5.3), title="Task-level paired screening")
    axes[1].invert_yaxis()
    axes[1].legend()
    fig.suptitle(f"BF16 {bf16.sum()}/50 | AWQ W4 {w4.sum()}/50 — matched state hashes and seeds\n40 new initial states; 10 repeats; not a complete implementation/performance certification")
    fig.tight_layout()
    fig.savefig(args.output / "baseline-gate.png", dpi=160)
    plt.close(fig)
    print(json.dumps(result["all_50"]))


if __name__ == "__main__":
    main()
