"""Export real frames and video hashes from the two completed stage-rescue runs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not (args.batch / "complete.json").is_file():
        raise SystemExit("Only completed batches")
    args.output.mkdir(parents=True, exist_ok=False)
    videos, frame_rows, runs = [], [], []
    for directory in sorted(args.batch.glob("stage-*")):
        text = (directory / "console.log").read_text()
        paths = re.findall(r"Saved rollout MP4 at path ([^\r\n]+)", text)
        evals = list(directory.glob("EVAL-*.txt"))
        if len(paths) != 10 or len(evals) != 1:
            raise SystemExit("Ten original videos and one evaluation required")
        successes = re.findall(r"^Success: (True|False)$", evals[0].read_text(), re.M)
        if len(successes) != 10: raise SystemExit("Incomplete evaluation")
        resolved = []
        for task, name in enumerate(paths):
            path = (Path("/root") / name).resolve()
            if Path("/root/rollouts") not in path.parents or not path.is_file():
                raise SystemExit("Missing or unsafe video: " + str(path))
            resolved.append(path)
            videos.append({"configuration": directory.name, "task_id": task,
                "success": successes[task] == "True", "source": str(path),
                "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        runs.append((directory.name, resolved, successes))
    for task in (5, 7):
        fig, axes = plt.subplots(len(runs), 6, figsize=(14, 5.8), squeeze=False)
        for row, (name, paths, successes) in enumerate(runs):
            cap = cv2.VideoCapture(str(paths[task]))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if count < 1: raise SystemExit("Unreadable original video")
            for col, requested in enumerate((0, 40, 80, 120, 160, 219)):
                index = min(requested, count-1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok: raise SystemExit("Unreadable video frame")
                axes[row, col].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                axes[row, col].set_title("frame {}".format(index), fontsize=9)
                axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
                if col == 0: axes[row, col].set_ylabel(name+"\n"+successes[task], fontsize=9)
                frame_rows.append({"task_id": task, "configuration": name, "requested_frame": requested,
                    "actual_frame": index, "video_frames": count, "source_video": str(paths[task])})
            cap.release()
        fig.suptitle("Task {}: original closed-loop video frames, shared initial state\nLater observations differ by policy; last frame held when the episode ended".format(task), fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, .9])
        fig.savefig(args.output / ("task-{}-stage-frames.png".format(task)), dpi=150)
        plt.close(fig)
    with (args.output / "video_frames.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(frame_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(frame_rows)
    (args.output / "video_manifest.json").write_text(json.dumps(videos, indent=2)+"\n")
    print(json.dumps({"videos": len(videos), "bytes": sum(r["bytes"] for r in videos)}, indent=2))


if __name__ == "__main__": main()
