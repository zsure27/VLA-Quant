"""One-time byte-preserving reports/results relocation plus task-oriented Chinese dossiers.

Dry run by default. Refuses symlinks/reparse points and conflicting destinations.
Never rewrites frozen evidence, scientific source, plans, runtime locks or raw results.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import stat
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "Ex_Reports/_legacy"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(path):
    p = path.lower()
    if "20261015-046-quant-finetuning" in p:
        return "20261015_普通与SVD初始化微调对照"
    if "20261016-091-residual" in p:
        if "b1" in p or "temporal" in p:
            return "20261017_训练取帧时间覆盖对照"
        if any(x in p for x in ("m0", "init-only", "first50", "reset_", "reset-", "cumulative-", "r0")):
            return "20261016_零步SVD补偿与直接量化对照"
        return "20261016_残差补偿研究规划与会话收尾"
    if "three-rounds-summary" in p or "progress-summary" in p or "/summaries/" in p:
        return "20260918_跨任务进展汇总与后续计划"
    if "b5" in p or "p5-contextual-routing" in p:
        return "20261012_上下文路由与专家选择"
    if "b4" in p:
        return "20261009_联合视觉语言微调"
    if "b2-b3" in p or "b2_b3" in p:
        return "20261005_视觉与语言补偿范围对照"
    if "smoothquant" in p or "secondary-smoothquant" in p:
        return "20260915_权重激活平滑量化探索"
    if "qvla-awq-alignment" in p:
        return "20261012_量化实现与官方基线对齐"
    if "quantization-finetuning-literature" in p or "research-plan" in p:
        return "20261011_量化微调文献与研究设计"
    if "position-weight2" in p or "flip-mechanism" in p or "failure-timeline" in p:
        return "20260928_动作误差与失败时序诊断"
    if "on-policy" in p or "onpolicy" in p or "student-state" in p or "distribution-triad" in p:
        return "20260927_学生状态分布与教师监督对齐"
    if "envseed" in p or "reset-preflight" in p or "validity-smoke" in p:
        return "20260929_环境种子与配对协议修复"
    if "data-contract" in p or "data-scope" in p or "registry-audit" in p or "/shared/" in p:
        return "20260925_数据角色模型身份与来源审计"
    if "router" in p or "p4-expert" in p or "/archive/" in p:
        return "20260916_专家互补与早期路由分析"
    if "p3-static-backbone" in p:
        return "20260923_混合位宽与静态主干筛查"
    if "response-svd" in p:
        return "20260917_响应SVD残差方向探索"
    if "p2-shared-peft" in p:
        return "20260925_共享LoRA与动作拟合探索"
    if "p0-foundation" in p:
        return "20260915_AWQ基线与视觉语言量化诊断"
    return "20260915_公共索引与历史辅助材料"


SUMMARIES = {
    "20261015_普通与SVD初始化微调对照": "正式各450回合：普通零输出初始化Z为2/450，Response-SVD初始化S为426/450，424次救回、0次新增失败。当前训练配方对初始化高度敏感；SVD另有校准与分解成本，不能据此认定普通LoRA普遍无效。",
    "20261016_零步SVD补偿与直接量化对照": "M0正式各450回合：PTQ77/450、S0 184/450，119次救回、12次新增失败，净增23.78个百分点。零优化步的补偿本身有效但不足以恢复全部能力；仍有校准与教师成本。重叠微测不增加分母。",
    "20261017_训练取帧时间覆盖对照": "B1正式各250回合：Early241/250（96.4%），Phase110/250（44%）；Phase相对Early 0次救回、131次新增失败，净低52.4个百分点。按用户要求结束比较，不补450；中断reset30–39保留但排除。结果表明当前取帧方案存在明显差异，物理原因尚未确定。",
    "20261012_上下文路由与专家选择": "保留B5历史实现、训练与开发记录。B5云端归档只有部分历史区间，不能称完整可恢复备份。Routing作为后续框架策略接口，原候选与旧活动计划不自动恢复；是否有效须以具体任务记录和有效配对为准。",
}


def candidates():
    tracked = subprocess.check_output(["git", "-c", "safe.directory=" + str(ROOT), "-C", str(ROOT), "ls-files", "-z"])
    result = {p for p in tracked.decode().split("\0") if p.startswith(("reports/", "results/"))}
    # Include extant small working evidence in exactly these two trees; never walk unrelated roots.
    for top in ("reports", "results"):
        for base, dirs, files in __import__("os").walk(ROOT / top, followlinks=False):
            base = Path(base)
            for name in dirs + files:
                item = base / name
                if item.is_symlink() or getattr(item.lstat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                    raise ValueError("Refuse reparse entry: " + str(item))
            for name in files:
                result.add((base / name).relative_to(ROOT).as_posix())
    return sorted(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if DEST.exists():
        raise ValueError("Migration destination exists; use saved manifest for verification, do not rerun")
    files = candidates()
    records = []
    # Git-normalized bytes are the portable source of truth; retain working-copy SHA too.
    index_lines = subprocess.check_output(["git", "-c", "safe.directory=" + str(ROOT), "-C", str(ROOT),
                                           "ls-files", "--stage", "-z"]).decode().split("\0")
    blobs = {line.split("\t", 1)[1]: line.split()[1] for line in index_lines if line}
    batch = subprocess.Popen(["git", "-c", "safe.directory=" + str(ROOT), "-C", str(ROOT), "cat-file", "--batch"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    for old in files:
        source, destination = ROOT / old, DEST / old
        if ROOT not in source.resolve().parents or ROOT not in destination.resolve().parents or destination.exists():
            raise ValueError("Unsafe source/destination")
        record = {"old_path": old, "new_path": destination.relative_to(ROOT).as_posix(),
                  "bytes": source.stat().st_size, "sha256": sha(source), "task": classify(old)}
        if old in blobs:
            batch.stdin.write((blobs[old] + "\n").encode()); batch.stdin.flush()
            header = batch.stdout.readline().decode().split()
            remaining = int(header[2]); h = hashlib.sha256()
            while remaining:
                block = batch.stdout.read(min(4 * 1024**2, remaining)); h.update(block); remaining -= len(block)
            batch.stdout.read(1)
            record.update(git_blob_sha1=blobs[old], git_blob_sha256=h.hexdigest(), git_bytes=int(header[2]))
        record["storage"] = ("local_only_ignored" if old not in blobs and
                             (Path(old).suffix in {".npz", ".pyc"} or "__pycache__" in old) else "public_git")
        records.append(record)
    batch.stdin.close(); batch.wait(timeout=30)
    print(json.dumps({"files": len(records), "bytes": sum(r["bytes"] for r in records),
                      "tasks": len({r["task"] for r in records}), "apply": args.apply}))
    if not args.apply:
        return
    groups = defaultdict(list)
    for record in records:
        source, destination = ROOT / record["old_path"], ROOT / record["new_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if sha(source) != record["sha256"]:
            raise ValueError("Source changed since admission")
        source.rename(destination)  # individual scoped file rename, never recursive deletion
        if sha(destination) != record["sha256"]:
            raise ValueError("Moved file digest mismatch")
        groups[record["task"]].append(record)
    root = ROOT / "Ex_Reports"
    index = {"schema_version": 1, "date": "2026-10-20", "files": records,
             "all_moved_files_sha256_verified": True,
             "line_endings": "sha256/bytes describe the original working copy; git_blob fields preserve portable Git bytes. Existing .gitattributes remains effective.",
             "classification": "Approximate historical research-task grouping; no rerun or new evidence"}
    index["publication_scope"] = {"public_git_files": sum(r["storage"] == "public_git" for r in records),
                                   "local_only_ignored_files": sum(r["storage"] == "local_only_ignored" for r in records)}
    (root / "migration-manifest.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    navigation = ["# 按任务阅读实验记录", "", "每个任务一个中文文件：先整合结果与分析，再放多轮日志。旧证据按原字节保存在 `_legacy`，没有重新运行实验。", "",
                  "日期取能够从旧目录确认的任务起始日；旧记录缺少可靠时分时不补造。新任务精确到分钟。", "",
                  "| 任务 | 证据文件数 |", "|---|---:|"]
    for task, rows in sorted(groups.items()):
        name = task.split("_", 1)[1]
        lines = ["# " + name, "", "任务开始日期：" + task[:8] + "（历史归类，时分未确认）", "",
                 "## 整合后的实验结果与分析", "", SUMMARIES.get(task, "以下为历史任务的粗分类整理。没有重新计算、追加实验或把未通过门禁的尝试计入成绩；以带来源和有效配对的分析为准。"), "",
                 "原始日志中的RUNNING、待续片或关机授权均是当时记录，不代表当前任务处于活动状态。不同suite、协议或跨轮分母不直接合并。", ""]
        docs = [r for r in rows if r["old_path"].startswith("reports/") and Path(r["old_path"]).suffix.lower() == ".md"]
        if task not in SUMMARIES and docs:
            preferred = sorted(docs, key=lambda r: (any(k in Path(r["old_path"]).name for k in ("ANALYSIS", "REPORT", "SUMMARY")), r["old_path"]))[-1]
            excerpt = (ROOT / preferred["new_path"]).read_text(encoding="utf-8-sig")[:1800]
            lines += ["### 历史分析摘录（请结合原文日期）", "", "来源：[" + preferred["old_path"] + "](" + preferred["new_path"].removeprefix("Ex_Reports/") + ")", "", "````text", excerpt, "````", ""]
        lines += ["## 多轮实验日志", ""]
        for row in docs:
            text = (ROOT / row["new_path"]).read_text(encoding="utf-8-sig")
            fence = "`" * max(4, max((len(x) for x in re.findall(r"`+", text)), default=0) + 1)
            lines += ["### " + row["old_path"], "", "[原始记录及附件路径](" + row["new_path"].removeprefix("Ex_Reports/") + ")", "",
                      "<details><summary>展开本轮完整文字记录</summary>", "", fence + "text", text, fence, "", "</details>", ""]
        lines += ["## 原始结果、参数与恢复证据", "", "| 文件 | 字节数 |", "|---|---:|"]
        for row in rows:
            lines += ["| [" + row["old_path"] + "](" + row["new_path"].removeprefix("Ex_Reports/") + ") | " + str(row["bytes"]) + " |"]
        (root / (task + ".md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
        navigation.append("| [" + name + "](" + task + ".md) | " + str(len(rows)) + " |")
    navigation += ["", "新任务规范与命令见[框架指南](../docs/INFRA_GUIDE_CN.md)。", "",
                   "大模型参数、原始观测/动作、视频和完整大日志使用[私有GitHub归档](https://github.com/zsure27/VLA-Quant-Archives)。此目录保存分析、小结果及恢复索引。", ""]
    (root / "README.md").write_text("\n".join(navigation), encoding="utf-8")
    for top in ("reports", "results"):
        # Preserve empty old dirs; only a navigation stub is newly written at the root.
        (ROOT / top / "README_CN.md").write_text("# 历史路径入口\n\n所有实验记录已整合到 [Ex_Reports](../Ex_Reports/README.md)。\n\n原字节证据：[" + top + "](../Ex_Reports/_legacy/" + top + ")。冻结脚本需要旧路径时，在服务器运行 `python -m qvla.infra legacy-links --apply`，不复制数据。\n", encoding="utf-8")


if __name__ == "__main__":
    main()
