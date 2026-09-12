"""重放既有 AWQ profile；分离局部参数误差与上游输入误差，不重新搜索。"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch

from probe import digest, initialize_readonly, load_trusted, metric, official_functions, predict, save_json
from qvla.awq_block import apply_block_scales, apply_llama_entry, tensor_tree

VARIANTS = ("bf16", "w2_scale", "w2_scale_clip", "w2_no_clip", "w2_full",
            "w2_protect_attention", "w2_protect_mlp", "w4_full")


def variant_policy(variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    return {"bits": 4 if variant == "w4_full" else 2,
            "scale": variant != "bf16",
            "clip": variant not in ("bf16", "w2_scale", "w2_no_clip"),
            "quantize": variant not in ("bf16", "w2_scale", "w2_scale_clip")}


@torch.no_grad()
def configure(block, prefix, variant, profiles, official):
    """调用者先恢复完整块；保护子模块时保留所有配对缩放，避免破坏坐标配对。"""
    policy = variant_policy(variant)
    if not policy["scale"]:
        return
    entries, meta = profiles[policy["bits"]]
    apply_block_scales(block, meta["block_scales"][prefix], official)
    for name, module in block.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if variant == "w2_protect_attention" and name.startswith("self_attn."):
            continue
        if variant == "w2_protect_mlp" and name.startswith("mlp."):
            continue
        entry = dict(entries[prefix + "." + name])
        if not policy["clip"]:
            entry["clip_max"] = None
        if policy["quantize"]:
            apply_llama_entry(module, entry, official, policy["bits"], meta["group_size"])
        elif entry.get("clip_max") is not None:
            # 与 apply_llama_entry 相同的 BF16 裁剪，仅省略随后的量化。
            clip = entry["clip_max"].to(module.weight.device, module.weight.dtype)
            module.weight.reshape(*clip.shape[:2], -1).clamp_(-clip, clip)


def compare(reference, candidate, start, count):
    if not 0 <= start < start + count <= reference.shape[1]:
        raise ValueError("动作 token 范围不合法")
    return {"all_tokens": metric(reference, candidate),
            "action_tokens": metric(reference[:, start:start + count], candidate[:, start:start + count])}


def replay(block, record):
    args, kwargs = tensor_tree(record, next(block.parameters()).device)
    return block(*args, **kwargs)[0].detach().cpu().clone()


def capture(bundle, paths, layers):
    cfg, model, head, proprio, processor = bundle
    records, actions, handles = {}, {}, []
    current = {}
    original = model._regression_or_discrete_prediction

    def prediction(*args, **kwargs):
        bound = inspect.signature(original).bind(*args, **kwargs).arguments
        from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
        current["layout"] = (int(bound["NUM_PATCHES"] + bound["NUM_PROMPT_TOKENS"]),
                             ACTION_DIM * NUM_ACTIONS_CHUNK)
        return original(*args, **kwargs)

    def hook(layer):
        def before(_module, args, kwargs):
            key = (current["sample"], layer)
            if key in records or kwargs.get("past_key_value") is not None or kwargs.get("use_cache", False):
                raise RuntimeError("要求每帧每块只执行一次且无 KV cache")
            if kwargs.get("output_attentions", False):
                raise RuntimeError("禁止切换 eager attention")
            records[key] = tensor_tree((args, kwargs), "cpu")
        return before

    model._regression_or_discrete_prediction = prediction
    try:
        for layer in layers:
            handles.append(model.language_model.model.layers[layer].register_forward_pre_hook(
                hook(layer), with_kwargs=True))
        for path in paths:
            current.clear()
            current["sample"] = path.name
            raw, normalized, hashes = predict(path, cfg, model, head, proprio, processor)
            actions[path.name] = {"raw": raw, "normalized": normalized,
                                  "hashes": hashes, "layout": current["layout"]}
            print("[capture]", path.name, flush=True)
    finally:
        model._regression_or_discrete_prediction = original
        for handle in handles:
            handle.remove()
    if len(records) != len(paths) * len(layers):
        raise RuntimeError("捕获块数量不匹配")
    return records, actions


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="10,11,12,25,26")
    parser.add_argument("--samples", default="35,48,57")
    args = parser.parse_args()
    layers = sorted(set(map(int, args.layers.split(","))))
    ids = sorted(set(map(int, args.samples.split(","))))
    if not layers or not ids or any(x not in range(32) for x in layers) or any(x not in range(32, 64) for x in ids):
        raise ValueError("仅允许本轮 32..63 诊断样本与 0..31 层")
    args.output.mkdir(parents=True, exist_ok=False)
    from qvla.runtime_contract import assert_oft_runtime
    from qvla.reproducibility import seed_all
    from qvla.run_eval_official_quant import load_profiles
    from qvla.baseline_contract import checkpoint_identity
    runtime = assert_oft_runtime()
    seed_all(7)
    official, sources = official_functions(args.official_root)
    identity = checkpoint_identity(args.checkpoint)
    profiles = {}
    paths = [args.base / "samples" / f"sample-{i:04d}.npz" for i in ids]
    teacher_manifest = json.loads((args.base / "teacher/manifest.json").read_text())
    old_w2_manifest = json.loads((args.base / "awq-w2-language/manifest.json").read_text())
    if old_w2_manifest["profiles"][0]["sha256"] != digest(args.base / "profiles/w2.pt"):
        raise ValueError("既有 W2 profile 已被替换")
    previous = {r["sample"]: r for r in json.loads((args.base / "awq-w2-language/metrics.json").read_text())}
    for bits in (2, 4):
        entries, meta = load_profiles([args.base / f"profiles/w{bits}.pt"], "awq")
        if meta["bits"] != bits or meta["checkpoint_identity"] != identity or meta["seed"] != 7:
            raise ValueError("profile 位宽、检查点或种子不匹配")
        if set(meta["block_scales"]) != {f"language_model.model.layers.{i}" for i in range(32)}:
            raise ValueError("block scales 不完整")
        for saved, loaded in (("awq_auto_scale", "awq_auto_scale_block"),
                              ("awq_auto_clip", "awq_auto_clip"), ("awq_quantizer", "awq_quantize")):
            if meta["official_sources"][saved]["sha256"] != sources[loaded]["sha256"]:
                raise ValueError("官方源码变化")
        for path in paths:
            if path.name in meta["sample_names"] or digest(path) != teacher_manifest["samples"][path.name]:
                raise ValueError("校准泄漏或诊断样本变化")
        profiles[bits] = (entries, meta)
    for key in ("sample_names", "sample_sha256", "group_size", "seed", "checkpoint_identity"):
        if profiles[2][1][key] != profiles[4][1][key]:
            raise ValueError(f"W2/W4 校准条件不匹配：{key}")
    bundle = initialize_readonly(args.checkpoint, 7)
    model = bundle[1]
    model.eval()
    model.language_model.config.use_cache = False
    bundle[2].eval()
    source_hash = digest(inspect.getsourcefile(type(model)))
    if model.language_model.config._attn_implementation != "sdpa":
        raise ValueError("必须 SDPA")
    if any(p[1]["model_source_sha256"] != source_hash for p in profiles.values()):
        raise ValueError("模型源码变化")
    for entries, meta in profiles.values():
        for name, module in model.named_modules():
            if name in entries and list(module.weight.shape) != entries[name]["shape"]:
                raise ValueError("profile 参数形状变化")
    if teacher_manifest["checkpoint_identity"] != identity or teacher_manifest["seed"] != 7:
        raise ValueError("教师身份变化")
    import transformers
    import timm
    for key, value in (("torch", torch.__version__), ("transformers", transformers.__version__),
                       ("timm", timm.__version__), ("model_source_sha256", source_hash), ("llm_attention", "sdpa")):
        if teacher_manifest[key] != value or old_w2_manifest[key] != value:
            raise ValueError(f"运行环境不匹配：{key}")
    if any(p.dtype != torch.bfloat16 for p in model.language_model.parameters()):
        raise ValueError("要求原语言模型参数为 BF16，不隐式改变精度")
    manifest = {"script_sha256": digest(__file__), "torch": torch.__version__,
                "runtime_sha256": runtime, "model_source_sha256": source_hash,
                "checkpoint_identity": identity, "sources": sources, "seed": 7,
                "samples": {p.name: digest(p) for p in paths}, "layers": layers,
                "profiles": {str(b): digest(args.base / f"profiles/w{b}.pt") for b in profiles},
                "variants": VARIANTS,
                "limits": "结果选择的三个诊断样本；局部干预，不是全模型保护或成功率；不更新校准参数"}
    save_json(args.output / "manifest.json", manifest)
    blocks = model.language_model.model.layers
    # 约一份语言模型大小的 CPU 快照；不复制整模型到 GPU。
    snapshots = [{k: v.detach().cpu().clone() for k, v in b.state_dict().items()} for b in blocks]
    results, controls = [], []
    try:
        teacher_records, teacher_actions = capture(bundle, paths, layers)
        for path in paths:
            saved = load_trusted(args.base / "teacher" / (path.stem + ".pt"))
            current = teacher_actions[path.name]
            error = metric(saved["normalized"], current["normalized"])["mse"]
            if saved["input_hashes"] != current["hashes"] or error > 1e-10:
                raise RuntimeError("既有教师复现失败，停止")
            controls.append({"sample": path.name, "teacher_repeat_mse": error})
        for i, block in enumerate(blocks):
            configure(block, f"language_model.model.layers.{i}", "w2_full", profiles, official)
        w2_records, w2_actions = capture(bundle, paths, layers)
        for index, path in enumerate(paths):
            t, q = teacher_actions[path.name], w2_actions[path.name]
            value = metric(t["normalized"], q["normalized"])["mse"]
            old = previous[path.name]["normalized_action"]["mse"]
            if t["hashes"] != q["hashes"] or t["layout"] != q["layout"] or abs(value - old) > 1e-8:
                raise RuntimeError("既有语言 W2 动作指标未复现，停止")
            controls[index]["language_w2_mse"] = value
        for block, snapshot in zip(blocks, snapshots):
            block.load_state_dict(snapshot)
        save_json(args.output / "controls.json", controls)
        for layer in layers:
            block = blocks[layer]
            prefix = f"language_model.model.layers.{layer}"
            refs = {}
            for path in paths:
                for context, records in (("teacher", teacher_records), ("w2_upstream", w2_records)):
                    record = records[path.name, layer]
                    refs[path.name, context] = replay(block, record)
                    # 相同输入重复运行控制；禁止悄悄放宽。
                    if metric(refs[path.name, context], replay(block, record))["mse"] > 1e-10:
                        raise RuntimeError("单块重复性失败")
            try:
                for variant in VARIANTS:
                    block.load_state_dict(snapshots[layer])
                    configure(block, prefix, variant, profiles, official)
                    for path in paths:
                        start, count = teacher_actions[path.name]["layout"]
                        for context, records in (("teacher", teacher_records), ("w2_upstream", w2_records)):
                            actual = replay(block, records[path.name, layer])
                            results.append({"sample": path.name, "layer": layer,
                                "variant": variant, "context": context,
                                "same_input_error": compare(refs[path.name, context], actual, start, count),
                                "vs_teacher_path": compare(refs[path.name, "teacher"], actual, start, count)})
                    save_json(args.output / "metrics.json", results)
                    print(f"[block-audit] layer={layer} variant={variant}", flush=True)
            finally:
                block.load_state_dict(snapshots[layer])
    finally:
        for block, snapshot in zip(blocks, snapshots):
            block.load_state_dict(snapshot)
    # 全部单块实验后再检查整模型恢复，防止累计修改。
    _, restored = capture(bundle, paths, layers)
    for path in paths:
        if metric(teacher_actions[path.name]["normalized"], restored[path.name]["normalized"])["mse"] > 1e-10:
            raise RuntimeError("实验结束后的教师恢复检查失败")
    save_json(args.output / "complete.json", {"status": "COMPLETE", "records": len(results),
              "teacher_restored": True, "note": "仅局部离线诊断，不代表量化修复成功"})
    print("AWQ_BLOCK_AUDIT: COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
