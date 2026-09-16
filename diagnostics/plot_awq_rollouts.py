"""Plot measured LIBERO outcomes, source video frames and query-time action traces."""
import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def read_eval(directory):
    files = list(directory.glob("EVAL-*.txt"))
    if len(files) != 1:
        raise ValueError(f"Expected one raw evaluation in {directory}")
    text = files[0].read_text()
    manifests = [json.loads(x) for x in re.findall(r"^EPISODE_MANIFEST (.+)$", text, re.M)]
    success = [x == "True" for x in re.findall(r"^Success: (True|False)$", text, re.M)]
    if len(manifests) != 10 or len(success) != 10 or "Episode error:" in text:
        raise ValueError(f"Incomplete or errored evaluation: {directory}")
    return manifests, success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/qvla-repro"))
    parser.add_argument("--repo", type=Path, default=Path("/root/VLA-Quant"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    old = [
        ("BF16", args.root / "eval/bf16-paired-smoke10-20260915-201949", args.root / "src/QVLA/openvla-oft"),
        ("AWQ W4", args.root / "eval/awq-w4-paired-smoke10-20260915-203534", args.repo),
        ("AWQ W2 tuned", args.root / "eval/awq-current-smoke10-20260915-194658", args.repo),
    ]
    runs = [(name, directory) for name, directory, _ in old]
    latest = args.root / "eval/LATEST_DIAGNOSTIC.txt"
    batch = Path(latest.read_text().strip()) if latest.exists() else None
    trace_runs = []
    if batch:
        for mode, label in [("bf16", "BF16 same loader"), ("w4", "W4 repeat + trace"),
                            ("language", "W2 language only"), ("vision", "W2 vision only")]:
            directory = batch / mode
            if (directory / "exit-code.txt").exists() and (directory / "exit-code.txt").read_text().strip() == "0":
                runs.append((label, directory))
                trace_runs.append((mode, label, directory))
    reference = None
    summary, rows = {}, []
    for label, directory in runs:
        manifests, successes = read_eval(directory)
        if reference is None:
            reference = manifests
        if manifests != reference:
            raise ValueError(f"Paired initial states/seeds differ: {label}")
        rows.append(successes)
        summary[label] = {"successes": sum(successes), "episodes": len(successes),
                          "per_task": successes, "source": str(directory), "paired_manifests_equal": True}
    fig, ax = plt.subplots(figsize=(10, 1.2 + len(rows) * 0.6))
    ax.imshow(np.array(rows), cmap=ListedColormap(["#d35e60", "#4baf81"]), vmin=0, vmax=1, aspect="auto")
    ax.set(xticks=range(10), xlabel="LIBERO Spatial task ID (one initial state each)",
           yticks=range(len(rows)), yticklabels=[f"{x}: {sum(y)}/10" for (x, _), y in zip(runs, rows)])
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            ax.text(j, i, "S" if value else "F", ha="center", va="center", color="white", weight="bold")
    ax.set_title("Paired closed-loop screening — S: success / F: failure\nMatched state hashes and seeds; repeats are not independent samples")
    fig.tight_layout()
    fig.savefig(args.output / "success-matrix.png", dpi=160)
    plt.close(fig)

    for task in (0, 9):
        fig, axes = plt.subplots(3, 6, figsize=(14, 7))
        for row, (label, directory, cwd) in enumerate(old):
            paths = re.findall(r"Saved rollout MP4 at path ([^\r\n]+)", (directory / "console.log").read_text())
            if len(paths) != 10:
                raise ValueError(f"Missing video paths for {label}")
            video = (cwd / paths[task]).resolve()
            capture = cv2.VideoCapture(str(video))
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if count < 1:
                raise ValueError(f"Cannot open {video}")
            for col, step in enumerate((0, 40, 80, 120, 160, 219)):
                frame_index = min(step, count - 1)
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Cannot read frame {frame_index} of {video}")
                axes[row, col].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                axes[row, col].set_title(f"frame {frame_index}", fontsize=9)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                if col == 0:
                    axes[row, col].set_ylabel(label)
            capture.release()
        fig.suptitle(f"Task {task}: real rollout frames (last frame held after episode ends)\nShared initial state; later observations diverge under each policy", fontsize=12)
        fig.tight_layout()
        fig.savefig(args.output / f"task-{task}-rollout-frames.png", dpi=150)
        plt.close(fig)

    traces = {}
    for mode, label, directory in trace_runs:
        records = [json.loads(line) for line in (directory / "policy-queries.jsonl").read_text().splitlines()]
        grouped = {}
        for record in records:
            grouped.setdefault(record["task"], []).append(record)
        traces[label] = grouped
        summary[label]["nonfinite_queries"] = sum(not r["finite"] for r in records)
    if traces:
        teacher = traces.get("BF16 same loader", {})
        state_space = next(iter(teacher.values()))[0].get("state_space", "normalized_proprio_after_get_action")
        for label, grouped in traces.items():
            errors = []
            for task, records in grouped.items():
                reference_record = teacher[task][0]
                if not np.allclose(records[0]["state"], reference_record["state"], atol=1e-10, rtol=0):
                    raise ValueError("First-query proprio differs")
                errors.append(float(np.abs(np.array(records[0]["raw_policy_chunk"]) - np.array(reference_record["raw_policy_chunk"])).mean()))
            summary[label]["first_query_action_l1_vs_bf16"] = float(np.mean(errors))
        for task_index in (0, 9):
            task = list(teacher)[task_index]
            fig, axes = plt.subplots(2, 2, figsize=(11, 7))
            teacher_first = np.array(teacher[task][0]["raw_policy_chunk"])
            teacher_state = np.array(teacher[task][0]["state"])
            for label, grouped in traces.items():
                records = grouped[task]
                states = np.array([r["state"] for r in records])
                chunks = np.concatenate([np.array(r["raw_policy_chunk"]) for r in records])
                first = np.array(records[0]["raw_policy_chunk"])
                if not np.allclose(teacher_state, states[0], atol=1e-10, rtol=0):
                    raise ValueError("First-query proprio differs; do not compare as shared-state action error")
                axes[0, 0].plot(states[:, 0], states[:, 1], marker=".", markersize=3, label=label)
                axes[0, 1].plot(states[:, 2], label=label)
                axes[1, 0].plot(chunks[:, -1], label=label, alpha=0.8)
                axes[1, 1].plot(np.sqrt(np.mean((first - teacher_first) ** 2, axis=1)), label=label)
                summary[label].setdefault("first_query_action_rmse", {})[str(task_index)] = float(np.sqrt(np.mean((first - teacher_first) ** 2)))
            prefix = "Normalized " if state_space.startswith("normalized") else "Raw "
            axes[0, 0].set(xlabel=prefix + "EEF x", ylabel=prefix + "EEF y", title="Query-time proprio components 0/1")
            axes[0, 1].set(xlabel="Policy query index", ylabel=prefix + "EEF z", title="Query-time proprio component 2")
            axes[1, 0].set(xlabel="Predicted action index", ylabel="Raw gripper command", title="Before gripper threshold/inversion")
            axes[1, 1].set(xlabel="Action index in first chunk", ylabel="RMSE vs same-loader BF16", title="Shared initial observation only")
            axes[0, 0].legend(fontsize=8)
            fig.suptitle(f"Task {task_index}: measured policy queries ({state_space})\nLater trajectories have different observations; their distance is not same-state quantization error", fontsize=11)
            fig.tight_layout()
            fig.savefig(args.output / f"task-{task_index}-action-diagnostics.png", dpi=160)
            plt.close(fig)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = ["method", "success_rate", "action_l1", "act_hidden_cosine", "average_bits", "model_size_gb",
              "peak_vram_gb", "latency_ms", "trainable_params", "first_query_action_l1_vs_bf16", "status", "notes"]
    rows = []
    for label, result in summary.items():
        rows.append({"method": label, "success_rate": result["successes"] / result["episodes"],
                     "trainable_params": 0, "first_query_action_l1_vs_bf16": result.get("first_query_action_l1_vs_bf16", ""),
                     "status": "paired_screening_complete_other_metrics_pending",
                     "notes": "10 matched initial states; simulated quantization held in BF16; blank means unmeasured; first-query teacher L1 is not trajectory or ground-truth action L1"})
    rows.extend([{"method": "AWQ W3A16", "status": "not_run", "notes": "No validated W3 profile/adapter contract"},
                 {"method": "AWQ packed W2/W4", "status": "not_run", "notes": "No validated packed implementation; cannot diagnose packing parity from fake quant results"}])
    for filename in ("baseline_precision.csv", "summary.csv"):
        with (args.output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({name: row["successes"] for name, row in summary.items()}))


if __name__ == "__main__":
    main()
