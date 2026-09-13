"""端到端语言 AWQ 干预计划；不修改输入 profile，不混用块内缩放坐标。"""
import re


def parse_layers(value):
    if not value:
        return set()
    layers = {int(x) for x in value.split(",")}
    if any(x not in range(32) for x in layers):
        raise ValueError("W4 层号必须为 0..31")
    return layers


def disable_clip(name, scope):
    if scope not in ("none", "all", "attention", "mlp"):
        raise ValueError("未知裁剪干预")
    return scope == "all" or (scope == "attention" and ".self_attn." in name) or (
        scope == "mlp" and ".mlp." in name)


def check_peer(entries, meta, other_entries, other_meta):
    if set(entries) != set(other_entries):
        raise ValueError("两个 profile 的目标范围不同")
    for key in ("checkpoint_identity", "sample_names", "sample_sha256", "group_size",
                "seed", "model_source_sha256", "llm_attention", "official_sources",
                "implementation_sources"):
        if meta[key] != other_meta[key]:
            raise ValueError(f"两个 profile 的校准条件不同：{key}")


def in_vision_branch(name, branch):
    prefixes = {"all": "vision_backbone.", "primary": "vision_backbone.featurizer.",
                "fused": "vision_backbone.fused_featurizer."}
    if branch not in prefixes:
        raise ValueError("未知视觉编码器分支")
    return name.startswith(prefixes[branch])


def vision_plan(entries, meta, bits, peer=None, branch="all"):
    """视觉保持原适配器的 scale/clip；语言不裁剪不能传播为视觉不裁剪。"""
    if bits not in (2, 4) or meta["bits"] != 2:
        raise ValueError("视觉组合只支持 W2/W4，基础语言 profile 必须 W2")
    if (bits == 4) != (peer is not None):
        raise ValueError("视觉 W4 必须独立 profile；视觉 W2 复用基础 profile")
    selected, metadata = peer if peer else (entries, meta)
    check_peer(entries, meta, selected, metadata)
    if metadata["bits"] != bits:
        raise ValueError("视觉 profile 位宽不符")
    targets = {n: (dict(e), bits, metadata["group_size"]) for n, e in selected.items()
               if n.startswith("vision_backbone.")}
    if len(targets) != 198:
        raise ValueError("视觉范围必须包含198个目标")
    targets = {n: v for n, v in targets.items() if in_vision_branch(n, branch)}
    if len(targets) != {"all": 198, "primary": 93, "fused": 105}[branch]:
        raise ValueError("视觉分支目标数不符")
    return targets


def plan(entries, meta, clip_scope, w4_layers, w4=None):
    """返回每块缩放来源、每个 Linear 的参数与实际位宽；仅用于语言 W2 干预。"""
    if meta["bits"] != 2 or clip_scope not in ("none", "all", "attention", "mlp"):
        raise ValueError("要求 W2 基础 profile")
    if bool(w4_layers) != (w4 is not None) or (w4_layers and clip_scope != "none"):
        raise ValueError("W4 层与 profile 必须同时提供，且不同时取消裁剪")
    if w4:
        e4, m4 = w4
        if m4["bits"] != 4 or set(e4) != set(entries):
            raise ValueError("W4 位宽或范围不符")
        for key in ("checkpoint_identity", "sample_names", "sample_sha256", "group_size",
                    "seed", "model_source_sha256", "llm_attention", "official_sources",
                    "implementation_sources"):
            if meta[key] != m4[key]:
                raise ValueError(f"W2/W4 校准条件不符：{key}")
    scales, targets, removed = {}, {}, []
    for layer in range(32):
        prefix = f"language_model.model.layers.{layer}"
        selected, metadata = w4 if layer in w4_layers else (entries, meta)
        scales[prefix] = metadata["block_scales"][prefix]
        names = [n for n in selected if n.startswith(prefix + ".")]
        if len(names) != 7:
            raise ValueError(f"不是完整的 7 个语言 Linear：{prefix}")
        for name in names:
            if not re.fullmatch(r"language_model\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)", name):
                raise ValueError(name)
            entry = dict(selected[name])
            if disable_clip(name, clip_scope):
                if entry.get("clip_max") is not None:
                    removed.append(name)
                entry["clip_max"] = None
            targets[name] = (entry, metadata["bits"], metadata["group_size"])
    return scales, targets, removed
