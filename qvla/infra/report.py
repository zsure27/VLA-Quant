"""One readable Chinese document per research task, spanning multiple rounds."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from .common import child, read_json, write_json


def task_filename(task):
    start = datetime.fromisoformat(task["started_at"].replace("Z", "+00:00"))
    return start.strftime("%Y%m%d_%H%M") + "_" + task["name_zh"] + ".md"


def record_round(repo, config, round_id, record):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", round_id):
        raise ValueError("Invalid round id")
    root = Path(repo) / "Ex_Reports"
    task_root = child(root / "_tasks", config["task"]["id"])
    task_root.mkdir(parents=True, exist_ok=True)
    identity = task_root / "task.json"
    if identity.exists() and read_json(identity) != config["task"]:
        raise ValueError("Task identity changed; use same task card across rounds")
    write_json(identity, config["task"])
    write_json(task_root / "runs" / (round_id + ".json"), record)
    render(root, config["task"])


def render(root, task):
    root = Path(root)
    task_root = child(root / "_tasks", task["id"])
    conclusion = task_root / "SUMMARY.md"
    summary = conclusion.read_text(encoding="utf-8") if conclusion.exists() else (
        "尚无经过审阅的模型实验结论。以下仅为执行记录；退出码或CPU框架测试通过不代表模型有效。\n")
    lines = ["# " + task["name_zh"], "", "开始时间：" + task["started_at"], "",
             "## 整合后的实验结果与分析", "", summary, "",
             "## 任务目的与比较条件", "", task["hypothesis"], "",
             "唯一变化：" + task["changed_variable"], "", "## 多轮运行日志", ""]
    for path in sorted((task_root / "runs").glob("*.json")):
        row = read_json(path)
        lines += ["### " + path.stem, "", "```json", __import__("json").dumps(row, ensure_ascii=False, indent=2), "```", ""]
    (root / task_filename(task)).write_text("\n".join(lines), encoding="utf-8")


def resolve_legacy(repo, old_path):
    """Resolve historical references without editing frozen source/JSON bytes."""
    if not old_path.startswith(("reports/", "results/")):
        return child(repo, old_path)
    return child(Path(repo) / "Ex_Reports" / "_legacy", old_path)
