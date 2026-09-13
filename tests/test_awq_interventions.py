"""不加载模型的路由测试：整块坐标配对、裁剪消融及 profile 不变性。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "diagnostics"))
from awq_interventions import plan, parse_layers, vision_plan


def fixture(bits):
    entries, scales = {}, {}
    for i in range(32):
        prefix = f"language_model.model.layers.{i}"
        scales[prefix] = f"scale-{bits}-{i}"
        for suffix in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                       "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            entries[prefix + "." + suffix] = {"clip_max": None if suffix in ("self_attn.q_proj", "self_attn.k_proj") else bits}
    meta = {k: "same" for k in ("checkpoint_identity", "sample_names", "sample_sha256", "seed",
            "model_source_sha256", "llm_attention", "official_sources", "implementation_sources")}
    meta.update(bits=bits, group_size=128, block_scales=scales)
    return entries, meta


class InterventionTest(unittest.TestCase):
    def test_vision_keeps_clipping_and_own_bits(self):
        e2, m2 = fixture(2)
        e4, m4 = fixture(4)
        for i in range(198):
            e2[f"vision_backbone.fixture{i}"] = {"clip_max": 2, "input_scale": "w2"}
            e4[f"vision_backbone.fixture{i}"] = {"clip_max": 4, "input_scale": "w4"}
        _, language, removed = plan(e2, m2, "all", set())
        visual = vision_plan(e2, m2, 4, (e4, m4))
        language.update(visual)
        self.assertEqual(len(language), 422)
        self.assertEqual(len(removed), 160)
        self.assertEqual(sum(v[1] == 4 for v in language.values()), 198)
        self.assertTrue(all(v[0]["clip_max"] == 4 and v[0]["input_scale"] == "w4" for v in visual.values()))
        self.assertTrue(all(v[0]["clip_max"] == 2 for v in vision_plan(e2, m2, 2).values()))
        with self.assertRaises(ValueError):
            vision_plan(e2, m2, 4)
        with self.assertRaises(ValueError):
            vision_plan(e2, m2, 2, (e4, m4))
        m4["seed"] = "changed"
        with self.assertRaises(ValueError):
            vision_plan(e2, m2, 4, (e4, m4))

    def test_clipping_counts_and_immutability(self):
        entries, meta = fixture(2)
        for scope, expected in (("all", 160), ("attention", 64), ("mlp", 96), ("none", 0)):
            _, targets, removed = plan(entries, meta, scope, set())
            self.assertEqual(len(targets), 224)
            self.assertEqual(len(removed), expected)
        self.assertEqual(entries["language_model.model.layers.0.mlp.up_proj"]["clip_max"], 2)

    def test_whole_block_w4(self):
        entries, meta = fixture(2)
        scales, targets, removed = plan(entries, meta, "none", {10, 11, 12}, fixture(4))
        self.assertEqual(sum(v[1] == 4 for v in targets.values()), 21)
        self.assertEqual(scales["language_model.model.layers.11"], "scale-4-11")
        self.assertEqual(scales["language_model.model.layers.13"], "scale-2-13")
        self.assertEqual(removed, [])

    def test_invalid_pairing(self):
        entries, meta = fixture(2)
        with self.assertRaises(ValueError):
            plan(entries, meta, "all", {11}, fixture(4))
        with self.assertRaises(ValueError):
            plan(entries, meta, "none", {11})
        e4, m4 = fixture(4)
        m4["seed"] = "different"
        with self.assertRaises(ValueError):
            plan(entries, meta, "none", {11}, (e4, m4))
        with self.assertRaises(ValueError):
            parse_layers("32")


if __name__ == "__main__":
    unittest.main()
