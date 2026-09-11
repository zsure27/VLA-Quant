"""先完整预检三份覆盖文件，再备份/安装；仅忽略 CRLF/LF 差异。"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile

# 由 QVLA 固定提交 26cc4821... 的 git blob 验证，不是 Windows 工作区字节哈希。
UPSTREAM_LF_SHA256 = {
    "prismatic/extern/hf/modeling_prismatic.py": "b5431a074c0025a12e46dc954a5e18d1d73477babb5ae42e3a12ab4b907f33a6",
    "experiments/robot/openvla_utils.py": "eed754d7c5f9821aae2fe0531dbe01df8c11df0d5c79b4aeeb9bb4452124bdf5",
    "experiments/robot/libero/run_libero_eval.py": "701a20b6ca2942aa36e0e7d567b25647268b936a23af79c5893534f10bec7802",
}


def normalized_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def plan_install(source, destination):
    plan = []
    for relative, upstream in UPSTREAM_LF_SHA256.items():
        src, dst = source / relative, destination / relative
        if not src.is_file() or not dst.is_file() or src.is_symlink() or dst.is_symlink():
            raise ValueError(f"文件缺失或为符号链接，停止安装：{relative}")
        actual, maintained = normalized_hash(dst), normalized_hash(src)
        if actual == maintained:
            continue  # 已安装的文件可以安全重跑，处理上次中途退出的情形。
        if actual != upstream:
            raise ValueError(f"未知源码修改，拒绝覆盖：{relative}\nexpected_LF={upstream} actual_LF={actual}")
        plan.append((src, dst, relative))
    return plan


def install(source, destination, backup_root, check_only=False):
    # 所有校验在第一次文件写入前完成，防止第二个文件失败后留下半套新代码。
    plan = plan_install(source, destination)
    if check_only:
        print(f"覆盖预检通过：待更新 {len(plan)} 个文件")
        return
    if not plan:
        print("覆盖文件已是维护版本，无需更改")
        return
    backup_root.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix="overlays-before-install-", dir=backup_root))
    for _, dst, relative in plan:
        saved = backup / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dst, saved)
    for src, dst, _ in plan:
        shutil.copy2(src, dst)
    print(f"覆盖安装完成；原文件可从此备份恢复：{backup}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--backup-root", type=Path, required=True)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    install(args.source.resolve(), args.destination.resolve(), args.backup_root.resolve(), args.check_only)


if __name__ == "__main__":
    main()
