"""直接使用固定版本官方 Llama block 搜索，保留每帧实际 mask/RoPE 上下文。

视觉分支另用显式标记的逐 Linear 适配，不能声称存在官方 OpenVLA ViT recipe。
不拼接不同帧的注意力上下文；只在官方损失的 token 维合并输出。
"""
from __future__ import annotations

import types
import torch


def tensor_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device).clone()
    if isinstance(value, dict):
        return {k: tensor_tree(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(tensor_tree(v, device) for v in value)
    if isinstance(value, list):
        return [tensor_tree(v, device) for v in value]
    if value is not None and not isinstance(value, (int, float, bool, str)):
        raise TypeError(f"不支持持久化的上下文 {type(value)}；禁止复用动态 KV cache")
    return value


@torch.no_grad()
def apply_block_scales(block, scales, official, features=None):
    modules = dict(block.named_modules())
    for previous, names, scale in scales:
        prev, layers = modules[previous], [modules[n] for n in names]
        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("AWQ block scale 非法")
        if isinstance(prev, torch.nn.Linear):
            if len(layers) != 1:
                raise ValueError("Linear→Linear 缩放必须只有一个后继")
            official["awq_scale_fc_fc"](prev, layers[0], scale)
        elif isinstance(prev, torch.nn.LayerNorm) or type(prev).__name__ == "LlamaRMSNorm":
            official["awq_scale_ln_fcs"](prev, layers, scale)
        else:
            raise TypeError(f"未审计的 AWQ 缩放前驱 {type(prev)}")
        if features is not None:
            for name in names:
                features[name].div_(scale.to(features[name].device, features[name].dtype))


@torch.no_grad()
def calibrate_llama_block(block, records, official, bits, group_size, clip_tokens=128):
    if type(block).__name__ != "LlamaDecoderLayer":
        raise TypeError("仅审计了 transformers 4.40.1 的 LlamaDecoderLayer")
    device = next(block.parameters()).device
    if device.type != "cuda":
        # 上游 org_sd={k:v.cpu()} 在 CPU 上会别名引用原权重，不能用于 CPU scale search。
        raise RuntimeError("官方 block 搜索须在 CUDA 上运行；CPU 只运行独立数值单测")
    linears = {n: m for n, m in block.named_modules() if isinstance(m, torch.nn.Linear)}
    if len(linears) != 7 or any(m.in_features % group_size for m in linears.values()):
        raise ValueError("Llama Linear 数量或 group_size 与基线不符")
    snapshot = {k: v.detach().cpu().clone() for k, v in block.state_dict().items()}
    features = {n: [] for n in linears}
    handles, attention_context, next_records, reference = [], [], [], []
    original_forward = block.self_attn.forward

    def capture_attention(_module, values, kwargs):
        attention_context.append(tensor_tree({k: v for k, v in kwargs.items() if k != "hidden_states"}, "cpu"))

    handles.append(block.self_attn.register_forward_pre_hook(capture_attention, with_kwargs=True))
    for name, layer in linears.items():
        handles.append(layer.register_forward_pre_hook(
            lambda _m, x, name=name: features[name].append(x[0].detach().cpu().clone())))
    try:
        for hidden, kwargs in records:
            hidden, kwargs = tensor_tree(hidden, device), tensor_tree(kwargs, device)
            if hidden.shape[0] != 1 or kwargs.get("past_key_value") is not None:
                raise ValueError("要求独立帧 batch=1、无 KV cache")
            output = block(hidden, **kwargs)[0]
            reference.append(output.detach().cpu())
            next_records.append((output.detach().cpu(), tensor_tree(kwargs, "cpu")))
    finally:
        for h in handles:
            h.remove()
    if len(attention_context) != len(records):
        raise RuntimeError("attention 上下文调用数不匹配")
    lengths = [r[0].shape[1] for r in records]
    joined = {n: torch.cat(xs, dim=1).to(device) for n, xs in features.items()}

    def replay_attention(_self, hidden_states, **_ignored):
        if hidden_states.shape[1] != sum(lengths):
            raise ValueError("官方搜索 token 布局发生变化")
        outputs = []
        for x, kwargs in zip(hidden_states.split(lengths, dim=1), attention_context):
            outputs.append(original_forward(x, **tensor_tree(kwargs, device))[0])
        return (torch.cat(outputs, dim=1),)

    try:
        block.self_attn.forward = types.MethodType(replay_attention, block.self_attn)
        scales = official["awq_auto_scale_block"](
            block, {}, bits, {"zero_point": True, "q_group_size": group_size}, joined)
        block.self_attn.forward = original_forward
        apply_block_scales(block, scales, official, joined)
        # 无量化的重参数化检查：失败则先排错，禁止保存为已校准 profile。
        equivalence = []
        for (x, kwargs), expected in zip(records, reference):
            actual = block(tensor_tree(x, device), **tensor_tree(kwargs, device))[0].float().cpu()
            error = ((actual - expected.float()).square().mean() /
                     expected.float().square().mean().clamp_min(1e-12)).item()
            equivalence.append(error)
        if max(equivalence) > 1e-3:
            raise RuntimeError(f"AWQ 无量化缩放等价性失败: {max(equivalence)}")
        entries = {}
        for name, layer in linears.items():
            clip = None
            if not any(s in name for s in ("q_", "k_", "query", "key", "Wqkv")):
                rows = joined[name].reshape(-1, joined[name].shape[-1]).float()
                clip = official["awq_auto_clip"](
                    layer.weight.float(), rows, n_bit=bits,
                    q_config={"zero_point": True, "q_group_size": group_size},
                    n_sample_token=min(clip_tokens, len(rows))).cpu().float()
            entries[name] = {"shape": list(layer.weight.shape), "clip_max": clip,
                             "recipe": "official_llama_block_v1", "module_type": "Linear"}
        return entries, [(p, list(ns), s.cpu().float()) for p, ns, s in scales], next_records, equivalence
    finally:
        block.self_attn.forward = original_forward
        block.load_state_dict(snapshot)


@torch.no_grad()
def apply_llama_entry(layer, entry, official, bits, group_size):
    weight = layer.weight.detach().clone()  # 保持官方 fake quant 的 BF16 数学路径
    clip = entry.get("clip_max")
    if clip is not None:
        clip = clip.to(weight.device, weight.dtype)
        grouped = weight.reshape(*clip.shape[:2], -1)
        grouped.clamp_(-clip, clip)
    weight = official["awq_quantize"](weight, n_bit=bits, zero_point=True, q_group_size=group_size)
    layer.weight.copy_(weight)
