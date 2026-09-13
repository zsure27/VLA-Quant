"""选择新 Spatial 验证轨迹：排除已知校准、诊断及历史 probe 使用记录。"""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, base, artifacts, output):
    source_rows = json.loads((source / "manifest.json").read_text(encoding="utf-8"))["samples"]
    base_rows = json.loads((base / "samples/manifest.json").read_text(encoding="utf-8"))["samples"]
    excluded_episodes = {(r["suite"], r["episode"]) for r in base_rows}
    excluded_hashes = set()
    history = []
    for row in base_rows:
        if Path(row["file"]).name != row["file"]:
            raise ValueError("非法基线样本文件名")
        value = sha(base / "samples" / row["file"])
        if value != row["sha256"]:
            raise ValueError("原基线样本哈希变化")
        excluded_hashes.add(value)
    # 旧 SQ/控制实验曾使用原始 0000..0007、0064..0071，保守排除。
    old_names = {f"sample-{i:04d}.npz" for i in list(range(8)) + list(range(64, 72))}
    for row in source_rows:
        if row["file"] in old_names:
            excluded_episodes.add((row["suite"], row["episode"]))
    # 仅识别 probe 顶层 samples={文件名:内容SHA256}，不能把全数据目录说明误当成使用记录。
    for path in sorted(artifacts.rglob("manifest.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        samples = data.get("samples")
        if isinstance(samples, dict) and samples and all(
            isinstance(v, str) and re.fullmatch(r"[0-9a-fA-F]{64}", v) for v in samples.values()
        ):
            excluded_hashes.update(v.lower() for v in samples.values())
            history.append({"file": str(path), "sha256": sha(path)})
    candidates = []
    for row in sorted(source_rows, key=lambda r: r["file"]):
        if row["suite"] != "libero_spatial_no_noops":
            continue
        if Path(row["file"]).name != row["file"]:
            raise ValueError("非法源样本文件名")
        value = sha(source / row["file"])
        if value in excluded_hashes:
            excluded_episodes.add((row["suite"], row["episode"]))
        candidates.append((row, value))
    selected, used_episodes, used_hashes = [], set(), set()
    for row, value in candidates:
        key = (row["suite"], row["episode"])
        if key in excluded_episodes or key in used_episodes or value in used_hashes or value in excluded_hashes:
            continue
        selected.append((row, value))
        used_episodes.add(key)
        used_hashes.add(value)
        if len(selected) == 32:
            break
    if len(selected) != 32:
        raise ValueError(f"排除已知使用轨迹后仅剩 {len(selected)} 条；停止，不退回旧诊断集")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for i, (row, value) in enumerate(selected):
        name = f"sample-{1000 + i:04d}.npz"
        shutil.copy2(source / row["file"], output / name)
        if sha(output / name) != value:
            raise ValueError("复制内容校验失败")
        rows.append({**row, "file": name, "source_file": row["file"], "sha256": value, "role": "validation"})
    manifest = {"samples": rows, "source": str(source), "base": str(base),
        "history_manifests": history, "excluded_trajectory_count": len(excluded_episodes),
        "selection": "按源文件名排序选择剩余32条，不按模型结果挑选；不重新校准",
        "limits": "仅保证与可核对的既有使用记录隔离；同套件一轨迹一帧，不代表独立任务或闭环成功率"}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SPATIAL_VALIDATION_SPLIT: PASS", "trajectories=32", output)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "base", "artifacts", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    prepare(args.source, args.base, args.artifacts, args.output)
