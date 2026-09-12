"""CPU 测试只验证干预路由/裁剪与复制语义，不冒充 GPU 官方 AWQ 验证。"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "diagnostics"))
import awq_block_audit as audit


class AuditTest(unittest.TestCase):
    def setUp(self):
        self.block = torch.nn.Module()
        self.block.self_attn = torch.nn.Module()
        self.block.self_attn.q_proj = torch.nn.Linear(4, 2, bias=False)
        self.block.mlp = torch.nn.Module()
        self.block.mlp.up_proj = torch.nn.Linear(4, 2, bias=False)
        self.prefix = "block"
        entries = {"block." + n: {"clip_max": torch.full((2, 1, 1), .25)}
                   for n, m in self.block.named_modules() if isinstance(m, torch.nn.Linear)}
        self.profiles = {b: (entries, {"block_scales": {"block": []}, "group_size": 4}) for b in (2, 4)}

    def test_no_clip_does_not_mutate_profile(self):
        with patch.object(audit, "apply_block_scales"), patch.object(audit, "apply_llama_entry") as apply:
            audit.configure(self.block, self.prefix, "w2_no_clip", self.profiles, {})
            self.assertEqual(apply.call_count, 2)
            self.assertTrue(all(c.args[1]["clip_max"] is None for c in apply.call_args_list))
        self.assertTrue(all(e["clip_max"] is not None for e in self.profiles[2][0].values()))

    def test_protection_and_bits(self):
        for variant, count, bits in (("bf16", 0, 2), ("w2_scale", 0, 2),
                                      ("w2_full", 2, 2), ("w4_full", 2, 4),
                                      ("w2_protect_attention", 1, 2), ("w2_protect_mlp", 1, 2)):
            with patch.object(audit, "apply_block_scales") as scale, patch.object(audit, "apply_llama_entry") as apply:
                audit.configure(self.block, self.prefix, variant, self.profiles, {})
                self.assertEqual(apply.call_count, count)
                self.assertEqual(scale.call_count, int(variant != "bf16"))
                for call in apply.call_args_list:
                    self.assertEqual(call.args[3], bits)
                if variant == "w2_protect_attention":
                    self.assertIs(apply.call_args.args[0], self.block.mlp.up_proj)
                if variant == "w2_protect_mlp":
                    self.assertIs(apply.call_args.args[0], self.block.self_attn.q_proj)

    def test_clip_without_quant(self):
        with torch.no_grad():
            for p in self.block.parameters():
                p.fill_(1)
        with patch.object(audit, "apply_block_scales"), patch.object(audit, "apply_llama_entry") as apply:
            audit.configure(self.block, self.prefix, "w2_scale_clip", self.profiles, {})
            self.assertEqual(apply.call_count, 0)
        self.assertTrue(all(torch.all(p == .25) for p in self.block.parameters()))

    def test_action_slice_and_context_clone(self):
        x = torch.ones(1, 5, 3)
        y = x.clone()
        y[:, 2:4] += 2
        result = audit.compare(x, y, 2, 2)
        self.assertEqual(result["action_tokens"]["mse"], 4)
        with self.assertRaises(ValueError):
            audit.compare(x, y, 4, 2)
        copied = audit.tensor_tree(((x,), {"use_cache": False}), "cpu")
        copied[0][0].zero_()
        self.assertTrue(torch.all(x == 1))


if __name__ == "__main__":
    unittest.main()
