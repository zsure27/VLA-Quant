"""不加载模型的路由测试：整块坐标配对、裁剪消融及 profile 不变性。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "diagnostics"))
from awq_interventions import plan, parse_layers, vision_plan, remove_primary_clips, primary_group_plan


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
    def test_primary_group_only(self):
        a, m = fixture(2)
        b, n = fixture(2)
        n['group_size'] = 64
        for i in range(93):
            a[f'vision_backbone.featurizer.{i}'] = {'clip_max': 1}
            b[f'vision_backbone.featurizer.{i}'] = {'clip_max': 2}
        targets = primary_group_plan(a, m, (b, n))
        self.assertEqual(len(targets), 93)
        self.assertTrue(all(v[1:] == (2,64) and v[0]['clip_max'] == 2 for v in targets.values()))
        self.assertEqual(m['group_size'], 128)
        self.assertEqual(n['group_size'], 64)
        n['seed'] = 'wrong'
        with self.assertRaises(ValueError):
            primary_group_plan(a,m,(b,n))

    def test_primary_no_clip_preserves_source(self):
        targets = {f"vision_backbone.featurizer.fixture{i}":
                   ({"clip_max": i if i else None, "input_scale": [1, 2]}, 2, 128) for i in range(93)}
        result, removed = remove_primary_clips(targets)
        self.assertEqual(len(removed), 92)
        self.assertTrue(all(v[0]["clip_max"] is None for v in result.values()))
        name = "vision_backbone.featurizer.fixture1"
        self.assertEqual(targets[name][0]["clip_max"], 1)
        self.assertEqual(result[name][0]["input_scale"], [1, 2])
        self.assertEqual(result[name][1:], (2, 128))
        with self.assertRaises(ValueError):
            remove_primary_clips({name: targets[name]})
        with self.assertRaises(ValueError):
            remove_primary_clips(result)

    def test_visual_branch_partition(self):
        entries, meta = fixture(2)
        for prefix, count in (("featurizer", 93), ("fused_featurizer", 105)):
            for i in range(count):
                entries[f"vision_backbone.{prefix}.fixture{i}"] = {"clip_max": 2}
        primary = vision_plan(entries, meta, 2, branch="primary")
        fused = vision_plan(entries, meta, 2, branch="fused")
        self.assertEqual(len(primary), 93)
        self.assertEqual(len(fused), 105)
        self.assertFalse(set(primary) & set(fused))
        self.assertEqual(set(primary) | set(fused), set(vision_plan(entries, meta, 2)))
        self.assertTrue(all(v[0]["clip_max"] == 2 for v in primary.values()))
        with self.assertRaises(ValueError):
            vision_plan(entries, meta, 2, branch="camera")

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
