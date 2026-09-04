"""小型 CPU 数值检查；在现有 AutoDL PyTorch 环境中运行。"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from probe import metric, official_functions, old_symmetric_row_quantize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--official-root", type=Path, required=True)
    args = p.parse_args()
    official, sources = official_functions(args.official_root)
    from qvla.official_quant_adapter import make_smoothquant_activation_hook, smooth_groups
    torch.manual_seed(7)
    w = torch.linspace(-1, 1, 128).reshape(1, 128)
    old = old_symmetric_row_quantize(w, 2, 1.0)
    assert old.unique().tolist() == [-1.0, 0.0, 1.0]
    new = official["awq_quantize"](w, n_bit=2, zero_point=True, q_group_size=128)
    assert len(new.unique()) == 4

    # 绕过量化时，实数 AWQ 缩放必须保持映射不变。
    x, w = torch.randn(3, 8), torch.randn(4, 8)
    s = torch.exp(torch.randn(8))
    torch.testing.assert_close(F.linear(x / s, w * s), F.linear(x, w))
    # 适配器采用此等效权重表达式，它不是“缩放相消导致量化失效”的错误。
    torch.testing.assert_close(F.linear(x, (w * s) / s), F.linear(x, w))

    norm = torch.nn.LayerNorm(8)
    linears = [torch.nn.Linear(8, 4), torch.nn.Linear(8, 6)]
    reference = [fc(norm(x)).detach().clone() for fc in linears]
    official["smooth_ln_fcs"](norm, linears, x.abs().amax(0), alpha=0.5)
    for fc, expected in zip(linears, reference):
        torch.testing.assert_close(fc(norm(x)), expected, rtol=1e-5, atol=1e-6)

    # 原始 SQ 函数会原地修改输入，适配器必须先克隆。
    value = torch.randn(1, 5, 8)
    backup = value.clone()
    linear = torch.nn.Linear(8, 4)
    hook = make_smoothquant_activation_hook(linear, official["smooth_activation_quantize"], 4)
    output = hook(linear, (value,))[0]
    torch.testing.assert_close(value, backup, rtol=0, atol=0)
    assert output.data_ptr() != value.data_ptr()
    # 官方函数未接收 t.view(...) 返回值，但这不会改变沿最后一维计算的语义。
    expected = official["smooth_activation_quantize"](value.clone().reshape(-1, 8), n_bits=4)
    torch.testing.assert_close(output, expected.reshape_as(value))

    names = set()
    for prefix, layers in (("vision_backbone.featurizer", 23), ("vision_backbone.fused_featurizer", 26)):
        names.add(prefix + ".patch_embed.proj")
        for i in range(layers):
            names.update(f"{prefix}.blocks.{i}.{suffix}" for suffix in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"))
    for i in range(32):
        names.update(f"language_model.model.layers.{i}.{suffix}" for suffix in (
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"))
    groups = smooth_groups(names)
    smoothed = {n for _, targets, _ in groups for n in targets}
    assert (len(names), len(groups), len(smoothed), len(names - smoothed)) == (422, 162, 258, 164)
    print(json.dumps({"status": "PASS", "old_w2_codes": old.unique().tolist(),
                      "official_w2_level_count": len(new.unique()), "sources": sources,
                      "scope": {"targets": 422, "norm_groups": 162, "smoothed_inputs": 258, "unsmoothed_inputs": 164}}, indent=2))


if __name__ == "__main__":
    main()
