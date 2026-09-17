"""Save a bounded VLA diagnostic session on the currently connected instance.

No network, credential reads, deletion, or shutdown. Use before vla_shutdown_remote.py.
Large model/calibration/profile originals remain on persistent storage; a manifest is
not their off-instance backup. This archives newly named diagnostics and code overlays.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
import subprocess
import tarfile


def run(*args):
    return subprocess.check_output(args, text=True)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_filter(info):
    if "__pycache__" in Path(info.name).parts or info.name.endswith((".pyc", ".pyo")):
        return None
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--result", type=Path, action="append", default=[])
    args = p.parse_args()
    if not args.label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.label):
        raise SystemExit("Use a simple session label")
    root = Path("/root/autodl-tmp/qvla-repro").resolve()
    repo = Path("/root/VLA-Quant").resolve()
    commit = run("git", "-C", str(repo), "rev-parse", "--short", "HEAD").strip()
    backup = root / "backups" / (args.label + "-" + datetime.now().strftime("%H%M%S") + "-" + commit)
    backup.mkdir(parents=True, exist_ok=False)
    (backup / "base-commit.txt").write_text(run("git", "-C", str(repo), "rev-parse", "HEAD"))
    (backup / "worktree.patch").write_text(run("git", "-C", str(repo), "diff", "HEAD", "--"))
    (backup / "git-status.txt").write_text(run("git", "-C", str(repo), "status", "--short"))
    (backup / "runtime-lock.txt").write_text(run("/root/miniconda3/envs/qvla-oft/bin/python", "-m", "pip", "freeze"))
    subprocess.check_call(["git", "-C", str(repo), "bundle", "create", str(backup / ("VLA-Quant-"+commit+".bundle")), "--all"])
    with tarfile.open(backup / "code-overlay.tar.gz", "w:gz") as tar:
        for name in ("language-stage-rescue", "closed-loop-stage-rescue", "language-family-rescue", "closed-loop-fixed-coordinates", "low-rank-recovery", "fp32-smoothing-pairs", "input-diag-recovery", "response-svd-recovery", "baseline-shards", "fp32-vision-smoothing"):
            path = root / "overlays" / name
            if path.is_dir(): tar.add(path, arcname="overlays/"+name, filter=source_filter)
        for name in ("diagnostics/export_checkpoint_manifest.py", "diagnostics/plot_awq_rollouts.py", "tests/test_awq_eval_scope.py"):
            path = repo / name
            if path.is_file(): tar.add(path, arcname="untracked-code/"+name)
    selected = []
    for value in args.result:
        path = value.resolve()
        if not any(root / name in path.parents for name in ("artifacts", "eval")) or not path.is_dir():
            raise SystemExit("Result must be a concrete existing artifact/eval directory")
        selected.append(path)
    videos = []
    video_root = Path("/root/rollouts").resolve()
    for directory in selected:
        if directory.parent != root / "eval": continue
        for console in directory.glob("*/console.log"):
            paths = re.findall(r"Saved rollout MP4 at path ([^\r\n]+)", console.read_text())
            command=(console.parent / "command.txt").read_text()
            trials=re.search(r"--num_trials_per_task\s+(\d+)",command)
            expected=10*int(trials.group(1)) if trials else 10
            if len(paths) != expected:
                raise SystemExit(f"{expected} original videos required for completed stage rollout")
            for name in paths:
                path = (Path("/root") / name).resolve()
                if video_root not in path.parents or not path.is_file():
                    raise SystemExit("Missing or unsafe rollout video: " + str(path))
                videos.append({"path": str(path), "bytes": path.stat().st_size,
                               "sha256": sha(path), "source_console": str(console)})
    (backup / "VIDEO_MANIFEST.json").write_text(json.dumps(videos, indent=2)+"\n")
    with tarfile.open(backup / "session-results.tar.gz", "w:gz") as tar:
        for path in selected:
            tar.add(path, arcname=path.relative_to(root).as_posix())
        for name in ("language-stage-rescue-a-launch.log", "language-stage-rescue-a-retry1-launch.log",
                     "language-stage-rescue-b-launch.log", "sq-controls-recheck-launch.log",
                     "sq-controls-negative-launch.log", "family-precision-launch.log"):
            path = root / "artifacts" / name
            if path.is_file(): tar.add(path, arcname="artifacts/"+name)
        path = root / "eval/language-stage-rescue10-launch.log"
        if path.is_file(): tar.add(path, arcname="eval/language-stage-rescue10-launch.log")
        for video in videos:
            path = Path(video["path"])
            tar.add(path, arcname="rollouts/"+path.relative_to(video_root).as_posix())
    large = []
    for name in ("artifacts/awq-spatial-20260912-163735-1136/profiles/w2.pt",
                 "artifacts/awq-spatial-20260912-163735-1136/profiles/w4.pt",
                 "artifacts/controls-20260911-234754-1133/profiles/sq/calibration.pt"):
        path = root / name
        if path.is_file(): large.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)})
    (backup / "LARGE_FILES_NOT_IN_GIT.json").write_text(json.dumps({
        "profiles": large, "model_root": str(root / "models"), "calibration_root": str(root / "calib"),
        "existing_validation_teacher": str(root / "artifacts/awq-validation-20260913-132827-1162/teacher"),
        "note": "These originals are not inside this archive or ordinary Git. Source clone is not independently audited here."}, indent=2)+"\n")
    (backup / "resume.json").write_text(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label, "hostname": run("hostname").strip(), "base_commit": commit,
        "result_directories": list(map(str,selected)), "repo": str(repo),
        "instructions": "Restore overlay without overwriting old evaluator; inspect completed metrics and read session report before new tests."}, indent=2)+"\n")
    files = sorted(p for p in backup.iterdir() if p.is_file())
    sums = "".join(sha(path)+"  "+path.name+"\n" for path in files)
    (backup / "SHA256SUMS.txt").write_text(sums)
    (backup / "RESULTS_SHA256SUMS.txt").write_text(sha(backup / "session-results.tar.gz")+"  session-results.tar.gz\n")
    print(str(backup), flush=True)


if __name__ == "__main__":
    main()
