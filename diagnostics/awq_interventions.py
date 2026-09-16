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


def remove_primary_clips(targets):
    """复制主视觉条目；只移除clip，不修改原profile或input_scale。"""
    if len(targets) != 93 or any(not in_vision_branch(n, "primary") or v[1] != 2 for n, v in targets.items()):
        raise ValueError("仅允许93个主视觉W2目标")
    result, removed = {}, []
    for name, (entry, bits, group) in targets.items():
        copied = dict(entry)
        if copied.get("clip_max") is not None:
            removed.append(name)
        copied["clip_max"] = None
        result[name] = (copied, bits, group)
    if not removed:
        raise ValueError("没有可移除裁剪，拒绝无效对照")
    return result, removed


def primary_group_plan(entries, meta, peer):
    """只路由独立G64 profile的主视觉；语言必须继续使用原G128。"""
    selected, other = peer
    if meta['bits'] != 2 or other['bits'] != 2 or meta['group_size'] != 128 or other['group_size'] != 64:
        raise ValueError('只支持主视觉W2 G128到G64对照')
    comparison = dict(other, group_size=128)
    check_peer(entries, meta, selected, comparison)
    targets = {n: (dict(e), 2, 64) for n, e in selected.items() if in_vision_branch(n, 'primary')}
    if len(targets) != 93:
        raise ValueError('主视觉必须93个目标')
    return targets


def plan(entries, meta, clip_scope, w4_layers, w4=None):
    """返回每块缩放来源、每个 Linear 的参数与实际位宽；仅用于语言 W2 干预。"""
    if meta["bits"] != 2 or clip_scope not in ("none", "all", "attention", "mlp"):
        raise ValueError("要求 W2 基础 profile")
    if bool(w4_layers) != (w4 is not None):
        raise ValueError("W4 层与 profile 必须同时提供")
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
            # clip_scope describes the W2 remainder. Rescued W4 blocks must keep
            # the clipping searched together with their own scales/coordinates.
            if layer not in w4_layers and disable_clip(name, clip_scope):
                if entry.get("clip_max") is not None:
                    removed.append(name)
                entry["clip_max"] = None
            targets[name] = (entry, metadata["bits"], metadata["group_size"])
    return scales, targets, removed


def current_w2_candidate_plan(entries, meta, primary_group64_peer):
    """共享的正式 W2 候选：语言 no-clip，DINO G64，SigLIP G128，视觉保留 clip。"""
    scales, targets, removed = plan(entries, meta, "all", set())
    visual = vision_plan(entries, meta, 2)
    visual.update(primary_group_plan(entries, meta, primary_group64_peer))
    targets.update(visual)
    if len(targets) != 422 or len(removed) != 160:
        raise ValueError("当前 W2 候选必须是422目标并移除160个语言裁剪")
    return scales, targets, removed


def language_stage_rollout_plan(entries, meta, w4_layers, peer):
    """Complete profile contract; rollout must select language-only scope.

    Vision entries are retained for contract validation, but not applied by the
    language-only evaluator. Rescued blocks keep their own W4 coordinates/clips.
    """
    if not w4_layers:
        raise ValueError("Stage rescue requires at least one W4 block")
    scales, targets, removed = plan(entries, meta, "all", w4_layers, peer)
    targets.update(vision_plan(entries, meta, 2))
    if len(targets) != 422:
        raise ValueError("Stage rollout requires the complete 422-target profile")
    return scales, targets, removed


def family_precision_plan(entries, meta, layers, family):
    """Isolate bit precision with every block retaining its original W2 coordinates.

    Selected attention/MLP linears use four bits with no extra clipping; this
    deliberately does not borrow a W4 profile or its differently scaled clips.
    It is a precision diagnostic, not a searched W4 baseline or PEFT method.
    """
    if not layers or family not in ("attention", "mlp"):
        raise ValueError("Family rescue requires layers and attention or mlp")
    scales, targets, removed = plan(entries, meta, "all", set())
    selected = []
    marker = ".self_attn." if family == "attention" else ".mlp."
    for name, (entry, bits, group) in targets.items():
        layer = int(name.split(".layers.")[1].split(".")[0])
        if layer in layers and marker in name:
            targets[name] = (entry, 4, group)
            selected.append(name)
    if len(selected) != len(layers)*(4 if family == "attention" else 3):
        raise ValueError("Incomplete precision rescue family")
    return scales, targets, removed
