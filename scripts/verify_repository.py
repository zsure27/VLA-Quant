"""对 GitHub 备份执行不依赖 GPU 的结构和安全检查。"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path


def repository_files(root: Path):
    """Yield files that can enter ordinary Git, excluding local complete archives."""
    excluded_roots = {root / ".git", root / "backups" / "experiments", root / ".local"}
    for path in root.rglob("*"):
        if any(parent == excluded or excluded in parent.parents for excluded in excluded_roots for parent in (path,)):
            continue
        if path.is_file():
            yield path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    targets = [x.strip() for x in (root / "configs/qvla-connected-422.txt").read_text().splitlines() if x.strip()]
    assert len(targets) == len(set(targets)) == 422
    assert sum(x.startswith("vision_backbone.featurizer.") for x in targets) == 93
    assert sum(x.startswith("vision_backbone.fused_featurizer.") for x in targets) == 105
    assert sum(x.startswith("language_model.") for x in targets) == 224
    skipped = []
    files = list(repository_files(root))
    python_paths = [path for path in files if path.suffix.lower() == ".py"]
    for path in python_paths:
        # 本地旧 Python 3.7 无法解析上游在 Python 3.8+ 使用的 starred return；
        # 新 AutoDL 的目标环境为 Python 3.10，会检查全部覆盖文件。
        if sys.version_info < (3, 8) and "overlays" in path.parts:
            skipped.append(str(path.relative_to(root)))
            continue
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    json_paths = [path for path in files if path.suffix.lower() == ".json"]
    for path in json_paths:
        json.loads(path.read_text(encoding="utf-8"))
    forbidden = re.compile(r"(github_pat_|ghp_[A-Za-z0-9]+|hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})")
    for path in files:
        if path.resolve() != Path(__file__).resolve() and path.suffix.lower() in {".py", ".sh", ".json", ".md", ".txt", ".csv"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            assert not forbidden.search(text), f"疑似密钥：{path}"
        if path.stat().st_size > 90 * 1024 * 1024:
            raise AssertionError(f"文件超过 90 MiB：{path}")
    print(f"仓库检查通过：targets=422, python={len(python_paths)}, json={len(json_paths)}, skipped={skipped}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"仓库检查失败：{error}", file=sys.stderr)
        raise
