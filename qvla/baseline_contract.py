"""可机器检查的基线合同：范围对齐不等于论文结果已复现。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

CONTRACT_ID = "qvla-oft-connected422-v1"


def canonical_targets():
    names = []
    for prefix, depth in (("vision_backbone.featurizer", 23),
                          ("vision_backbone.fused_featurizer", 26)):
        names.append(prefix + ".patch_embed.proj")
        for i in range(depth):
            names.extend(f"{prefix}.blocks.{i}.{s}" for s in
                         ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"))
    for i in range(32):
        names.extend(f"language_model.model.layers.{i}.{s}" for s in
                     ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                      "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"))
    return sorted(names)


def check_scope(names, complete=True):
    names = list(names)
    expected = set(canonical_targets())
    if len(names) != len(set(names)) or set(names) - expected:
        raise ValueError("量化范围含重复、排除模块或未知模块")
    if complete and set(names) != expected:
        raise ValueError(f"必须精确覆盖 422 个动作连通目标，缺少 {len(expected - set(names))} 个")


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def checkpoint_identity(path):
    """完整权重与动作头哈希；路径可迁移，但内容不能悄悄替换。"""
    root = Path(path)
    files = sorted(p for p in root.rglob("*") if p.is_file() and
                   ".cache" not in p.parts and p.suffix in
                   {".safetensors", ".bin", ".pt", ".pth", ".json", ".py", ".model"})
    if not any(p.suffix in {".safetensors", ".bin"} for p in files):
        raise ValueError("检查点目录没有模型权重")
    hashes = {p.relative_to(root).as_posix(): file_hash(p) for p in files}
    return {"files": hashes, "sha256": hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def sample_identity(paths):
    return {p.name: file_hash(p) for p in paths}
