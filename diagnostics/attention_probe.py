"""旁路重建少量真实 SDPA 概率，不切 eager，不改变策略前向的计算结果。"""
import math
import torch
import torch.nn.functional as F


def probabilities(q, k, query_ids, mask=None, scale=None, is_causal=False):
    if is_causal or q.shape[0] != 1 or q.shape[1] != k.shape[1]:
        raise ValueError("要求 OFT 双向 attention、batch=1，且 KV 已完成 GQA 展开")
    scores = q[:, :, query_ids].float() @ k.float().transpose(-2, -1)
    scores *= scale if scale is not None else 1 / math.sqrt(q.shape[-1])
    if mask is not None:
        if mask.shape[-2] != 1:
            mask = mask[..., query_ids, :]
        scores = scores.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else scores + mask.float()
    result = scores.softmax(-1)
    if not torch.isfinite(result).all():
        raise ValueError("attention 概率出现非有限值")
    return result


class AttentionTap:
    def __init__(self, model, recorder, layers):
        self.recorder, self.model = recorder, model
        self.active, self.data, self.handles = None, {}, []
        self.original = F.scaled_dot_product_attention
        for i in layers:
            module = model.language_model.model.layers[i].self_attn
            if type(module).__name__ != "LlamaSdpaAttention":
                raise ValueError("只能探测已验证的 SDPA 路径")
            def enter(_m, _x, i=i):
                self.active = str(i)
            def leave(_m, _x, _y):
                self.active = None
            self.handles.append(module.register_forward_pre_hook(enter))
            self.handles.append(module.register_forward_hook(leave))

        def observed(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
            output = self.original(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                                   is_causal=is_causal, **kwargs)
            if self.active is not None:
                if dropout_p != 0 or kwargs.get("enable_gqa", False):
                    raise ValueError("不支持训练 dropout 或未展开的 GQA")
                start, count = recorder.action_start, recorder.action_count
                if start is None or start + count > q.shape[-2]:
                    raise ValueError("动作读出切片无效")
                ids = torch.linspace(start, start + count - 1, min(count, 8), device=q.device).round().long().unique()
                p = probabilities(q, k, ids, attn_mask, kwargs.get("scale"), is_causal)
                self.data[self.active] = {"p": p.detach().cpu(), "query_ids": ids.cpu()}
            return output
        F.scaled_dot_product_attention = observed

    def reset(self):
        self.data, self.active = {}, None

    def compare(self, reference):
        if set(reference) != set(self.data):
            raise ValueError("teacher/candidate attention 层不匹配")
        output = {}
        patches = self.model.vision_backbone.get_num_patches()
        start, count = self.recorder.action_start, self.recorder.action_count
        ranges = {"image_main": (1, 1 + patches), "image_wrist": (1 + patches, 1 + 2 * patches),
                  "proprio": (1 + 2 * patches, 2 + 2 * patches), "text": (2 + 2 * patches, start),
                  "action_readout": (start, start + count)}
        for layer, item in self.data.items():
            r, q = reference[layer]["p"].float(), item["p"].float()
            if r.shape != q.shape or not torch.equal(reference[layer]["query_ids"], item["query_ids"]):
                raise ValueError("attention 位置/形状不匹配")
            mix = (r + q) / 2
            js = .5 * (r * (r.clamp_min(1e-12).log() - mix.clamp_min(1e-12).log()) +
                       q * (q.clamp_min(1e-12).log() - mix.clamp_min(1e-12).log())).sum(-1)
            output[layer] = {"js_mean": js.mean().item(), "js_per_head": js.mean((0, 2)).tolist(),
                "entropy_mean": (-(q * q.clamp_min(1e-12).log()).sum(-1)).mean().item(),
                "modality_mass": {name: {"teacher": r[..., lo:hi].sum(-1).mean().item(),
                                          "candidate": q[..., lo:hi].sum(-1).mean().item()}
                                  for name, (lo, hi) in ranges.items()}}
        return output

    def close(self):
        F.scaled_dot_product_attention = self.original
        for handle in self.handles:
            handle.remove()
