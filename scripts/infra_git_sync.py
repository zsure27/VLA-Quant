"""Publish only the current task's small, credential-scanned report files; no git add -A."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qvla.infra.common import read_json, write_json, now
from qvla.infra.report import task_filename


def git(*argv):
    return subprocess.check_output(["git", "-c", "safe.directory=" + str(ROOT), "-C", str(ROOT), *argv],
                                   text=True, encoding="utf-8", stderr=subprocess.PIPE).strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-id", required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--branch", default="main")
    p.add_argument("--include", action="append", default=[], help="An explicitly reviewed changed code/config file to publish")
    a = p.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", a.task_id) or not re.fullmatch(r"[a-zA-Z0-9_/-]+", a.branch):
        raise ValueError("Invalid task/branch")
    task_root = ROOT / "Ex_Reports/_tasks" / a.task_id
    task = read_json(task_root / "task.json")
    files = [ROOT / "Ex_Reports" / task_filename(task)] + sorted(task_root.rglob("*"))
    for relative in a.include:
        candidate = (ROOT / relative).resolve(strict=True)
        if ROOT not in candidate.parents or candidate.relative_to(ROOT).parts[0] not in {
                "qvla", "scripts", "configs", "docs", "diagnostics", "overlays", "tests"}:
            raise ValueError("Explicit code publication path outside allowed source directories")
        files.append(candidate)
    paths = []
    marker = re.compile(rb"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{25,}")
    for file in files:
        if not file.is_file():
            continue
        if file.is_symlink() or file.stat().st_size > 2 * 1024**2 or file.suffix not in (".md", ".json", ".csv", ".py", ".sh", ".txt"):
            raise ValueError("Publish only small text task evidence; raw material goes to private Release")
        if marker.search(file.read_bytes()):
            raise ValueError("Credential marker in publication scope")
        paths.append(file.relative_to(ROOT).as_posix())
    if git("diff", "--cached", "--name-only"):
        raise ValueError("Index already contains changes; do not mix another task's work")
    git("add", "--", *paths)
    if git("diff", "--cached", "--name-only"):
        git("commit", "-m", "Record experiment task " + a.task_id)
    head = git("rev-parse", "HEAD")
    # Fixed approved public code destination. Never print credential-bearing origin URLs.
    url = "https://github.com/zsure27/VLA-Quant.git"
    git("push", url, "HEAD:refs/heads/" + a.branch)
    remote = git("ls-remote", url, "refs/heads/" + a.branch).split()[0]
    if remote != head:
        raise ValueError("Remote commit verification failed")
    write_json(a.run_dir / "git-sync-receipt.json", {"time_utc": now(), "commit": head, "remote_verified": True})


if __name__ == "__main__":
    main()
