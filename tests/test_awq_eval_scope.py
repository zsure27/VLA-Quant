import unittest
from unittest.mock import patch

import torch

from qvla.baseline_contract import canonical_targets
from qvla.run_eval_official_quant import apply_quantization, select_awq_scope


class ScopeTest(unittest.TestCase):
    def test_scopes_partition_complete_contract(self):
        names = list(canonical_targets())
        language = select_awq_scope(names, "language")
        vision = select_awq_scope(names, "vision")
        self.assertEqual((len(language), len(vision)), (224, 198))
        self.assertEqual(set(language) | set(vision), set(names))
        self.assertFalse(set(language) & set(vision))
        with self.assertRaises(ValueError):
            select_awq_scope(names[:-1], "vision")

    def test_vision_does_not_apply_language_scales(self):
        names = list(canonical_targets())
        entries = {n: {"shape": (1, 1), "recipe": "official_llama_block_v1"} for n in names}
        metadata = {"bits": 2, "group_size": 128,
                    "block_scales": {f"language_model.model.layers.{i}": [] for i in range(32)}}
        model = torch.nn.Linear(1, 1)
        modules = {n: model for n in names}
        with patch("qvla.run_eval_official_quant.validate_targets", return_value=modules), \
             patch("qvla.run_eval_official_quant.apply_awq_entry") as vision_apply, \
             patch("qvla.awq_block.apply_llama_entry") as language_apply, \
             patch("qvla.awq_block.apply_block_scales") as scales:
            apply_quantization(model, entries, metadata, "awq", {}, 2, 16, awq_scope="vision")
            self.assertEqual(vision_apply.call_count, 198)
            language_apply.assert_not_called()
            scales.assert_not_called()

    def test_none_leaves_weights_untouched(self):
        model = torch.nn.Linear(3, 2)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        entries = {n: {} for n in canonical_targets()}
        self.assertEqual(apply_quantization(model, entries, {}, "awq", {}, 2, 16, awq_scope="none"), [])
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))


if __name__ == "__main__":
    unittest.main()
