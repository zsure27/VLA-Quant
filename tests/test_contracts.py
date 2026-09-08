"""CPU 回归测试；数值通过不等于完整 VLA/机器人评估通过。"""
import argparse
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvla.baseline_contract import canonical_targets, check_scope, checkpoint_identity
from qvla.reproducibility import seed_all, episode_seeds
from qvla.official_quant_adapter import load_official_functions, make_smoothquant_activation_hook, smooth_groups
from qvla.awq_block import apply_block_scales, tensor_tree

OFFICIAL = None


class ContractTests(unittest.TestCase):
    def test_exact_scope(self):
        names = canonical_targets()
        check_scope(names)
        checked = (Path(__file__).resolve().parents[1] / "configs/qvla-connected-422.txt").read_text().splitlines()
        self.assertEqual(set(checked), set(names))
        with self.assertRaises(ValueError):
            check_scope(names[:-1] + ["projector.fc1"])
        with self.assertRaises(ValueError):
            check_scope(names[:-1] + [names[0]])
        self.assertEqual(len(smooth_groups(set(names))), 162)

    def test_seed_streams(self):
        seed_all(7)
        a = (random.random(), np.random.rand(), torch.rand(3))
        seed_all(7)
        b = (random.random(), np.random.rand(), torch.rand(3))
        self.assertEqual(a[:2], b[:2])
        torch.testing.assert_close(a[2], b[2], rtol=0, atol=0)
        x = episode_seeds(7, 0, "spatial", 0, 0)
        self.assertEqual(x, episode_seeds(7, 0, "spatial", 0, 0))
        self.assertNotEqual(x, episode_seeds(7, 0, "spatial", 0, 1))
        self.assertNotEqual(x[0], x[1])

    def test_checkpoint_content_not_path(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            for folder in (a, b):
                (Path(folder) / "model.safetensors").write_bytes(b"fixture-only")
            self.assertEqual(checkpoint_identity(a), checkpoint_identity(b))
            (Path(b) / "model.safetensors").write_bytes(b"changed")
            self.assertNotEqual(checkpoint_identity(a), checkpoint_identity(b))

    def test_tensor_tree_no_alias(self):
        x = {"position_ids": torch.arange(5)}
        y = tensor_tree(x, "cpu")
        y["position_ids"].zero_()
        self.assertEqual(x["position_ids"].tolist(), list(range(5)))
        with self.assertRaises(TypeError):
            tensor_tree(object(), "cpu")

    def test_sq_forbids_silent_gradient_cut(self):
        layer = torch.nn.Linear(8, 4)
        hook = make_smoothquant_activation_hook(layer, OFFICIAL["smooth_activation_quantize"], 4)
        x = torch.randn(2, 8, requires_grad=True)
        with self.assertRaises(RuntimeError):
            hook(layer, (x,))
        with torch.no_grad():
            before = x.clone()
            y = hook(layer, (x,))[0]
            torch.testing.assert_close(before, x, rtol=0, atol=0)
            self.assertNotEqual(x.data_ptr(), y.data_ptr())

    def test_official_llama_reparameterization(self):
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        config = LlamaConfig(hidden_size=32, intermediate_size=64, num_attention_heads=4,
                             num_key_value_heads=4, attention_dropout=0.0)
        config._attn_implementation = "sdpa"
        block = LlamaDecoderLayer(config, 0).eval()
        seed_all(31)
        x = torch.randn(1, 7, 32)
        kwargs = {"attention_mask": torch.zeros(1, 1, 7, 7), "position_ids": torch.arange(7)[None],
                  "use_cache": False, "cache_position": torch.arange(7)}
        with torch.no_grad():
            before = block(x, **kwargs)[0]
            scales = [
                ("input_layernorm", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], torch.exp(torch.randn(32) * .3)),
                ("self_attn.v_proj", ["self_attn.o_proj"], torch.exp(torch.randn(32) * .3)),
                ("post_attention_layernorm", ["mlp.gate_proj", "mlp.up_proj"], torch.exp(torch.randn(32) * .3)),
                ("mlp.up_proj", ["mlp.down_proj"], torch.exp(torch.randn(64) * .3)),
            ]
            apply_block_scales(block, scales, OFFICIAL)
            after = block(x, **kwargs)[0]
        torch.testing.assert_close(before, after, rtol=2e-5, atol=2e-6)

    def test_profile_v2_and_linear_rejected(self):
        from qvla.run_eval_official_quant import load_profiles
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old.pt"
            torch.save({"format_version": 2, "method": "awq"}, path)
            with self.assertRaises(RuntimeError):
                load_profiles([path], "awq")
            torch.save({"format_version": 3, "method": "awq", "contract_id": "qvla-oft-connected422-v1",
                        "checkpoint_identity": {"sha256": "fixture"}, "sample_sha256": {"sample": "fixture"},
                        "algorithm": "linear_diagnostic_NOT_official_block"}, path)
            with self.assertRaisesRegex(RuntimeError, "逐 Linear"):
                load_profiles([path], "awq")

    def test_attention_uses_real_mask_and_scale(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "diagnostics"))
        from attention_probe import probabilities
        q, k, v = (torch.randn(1, 4, 8, 16) for _ in range(3))
        mask = torch.zeros(1, 1, 8, 8)
        mask[..., -1] = float("-inf")
        ids = torch.arange(8)
        p = probabilities(q, k, ids, mask, scale=.2)
        actual = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=.2)
        torch.testing.assert_close(p @ v, actual, atol=1e-6, rtol=1e-5)
        self.assertEqual(p[..., -1].abs().max().item(), 0)
        with self.assertRaises(ValueError):
            probabilities(q, k, ids, mask, is_causal=True)

    def test_attention_tap_does_not_change_forward(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "diagnostics"))
        from attention_probe import AttentionTap
        from types import SimpleNamespace
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        config = LlamaConfig(hidden_size=32, intermediate_size=64, num_attention_heads=4,
                             num_key_value_heads=4, attention_dropout=0.0)
        config._attn_implementation = "sdpa"
        block = LlamaDecoderLayer(config, 0).eval()
        model = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=[block])))
        recorder = SimpleNamespace(action_start=3, action_count=3)
        x = torch.randn(1, 7, 32)
        kwargs = {"attention_mask": torch.zeros(1, 1, 7, 7), "position_ids": torch.arange(7)[None],
                  "use_cache": False, "cache_position": torch.arange(7)}
        original = torch.nn.functional.scaled_dot_product_attention
        with torch.no_grad():
            before = block(x, **kwargs)[0]
            tap = AttentionTap(model, recorder, [0])
            try:
                after = block(x, **kwargs)[0]
                self.assertIn("0", tap.data)
            finally:
                tap.close()
        self.assertIs(original, torch.nn.functional.scaled_dot_product_attention)
        torch.testing.assert_close(before, after, atol=0, rtol=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", required=True, type=Path)
    args, remaining = parser.parse_known_args()
    OFFICIAL = load_official_functions(args.official_root)
    unittest.main(argv=[sys.argv[0]] + remaining)
