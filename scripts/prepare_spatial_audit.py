"""从现有样本选择不同 Spatial 轨迹，复制到新目录；不覆盖原始样本。"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.source / "manifest.json").read_text())
    rows = [r for r in manifest["samples"] if r["suite"] == "libero_spatial_no_noops"]
    rows = sorted(rows, key=lambda r: r["file"])[:64]
    if len(rows) != 64 or len({r["episode"] for r in rows}) != 64:
        raise ValueError("需要 64 条不同 Spatial 轨迹")
    hashes = []
    for row in rows:
        name = row["file"]
        if Path(name).name != name:
            raise ValueError("样本文件名必须不包含目录")
        hashes.append(hashlib.sha256((args.source / name).read_bytes()).hexdigest())
    if len(set(hashes)) != 64:
        raise ValueError("存在重复样本内容")
    args.output.mkdir(parents=True, exist_ok=False)
    selected = []
    for i, (row, digest) in enumerate(zip(rows, hashes)):
        name = f"sample-{i:04d}.npz"
        shutil.copy2(args.source / row["file"], args.output / name)
        selected.append({**row, "file": name, "source_file": row["file"],
                         "index": i, "sha256": digest,
                         "role": "calibration" if i < 32 else "diagnostic"})
    result = {"samples": selected, "source": str(args.source.resolve()),
              "selection": "按原文件名排序；前32校准、后32诊断；不是随机抽样或最终测试集",
              "limits": "每轨迹一帧，来自已有128条Spatial轨迹；不等于论文校准协议"}
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SPATIAL_SPLIT: PREPARED", args.output)


if __name__ == "__main__":
    main()
