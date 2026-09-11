"""无 GPU 测试：检查消融边界和整组选择。"""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "diagnostics"))
from smoothing_selection import excluded_norm


class SelectionTest(unittest.TestCase):
    def test_counts(self):
        names = [f"language_model.model.layers.{i}.{kind}" for i in range(32)
                 for kind in ("input_layernorm", "post_attention_layernorm")]
        for selection, expected in (("all", 0), ("no-late-attention", 9),
                                    ("no-late-mlp", 9), ("no-late-both", 18)):
            self.assertEqual(sum(excluded_norm(n, selection) for n in names), expected)

    def test_vision_untouched(self):
        self.assertFalse(excluded_norm("vision_backbone.featurizer.blocks.23.norm1", "no-late-both"))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            excluded_norm("anything", "invalid")


if __name__ == "__main__":
    unittest.main()
