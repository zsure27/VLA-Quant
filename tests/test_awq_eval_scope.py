import unittest
from unittest.mock import patch

import torch
from collections import Counter

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

    def test_stage_rollout_keeps_w4_coordinates_and_vision_bf16(self):
        from diagnostics.awq_interventions import language_stage_rollout_plan
        from tests.test_awq_interventions import fixture
        entries, metadata = fixture(2)
        peer_entries, peer_meta = fixture(4)
        for name in canonical_targets():
            for value, bits in ((entries, 2), (peer_entries, 4)):
                if name not in value: value[name] = {"clip_max": bits}
                value[name]["shape"] = (1, 1)
                value[name]["recipe"] = "official_llama_block_v1"
                clip = value[name]["clip_max"]
                value[name]["clip_max"] = None if clip is None else torch.tensor(float(clip))
        rescue = language_stage_rollout_plan(entries, metadata, set(range(8,16)), (peer_entries,peer_meta))
        self.assertEqual(len(rescue[1]), 422)
        self.assertEqual(len(rescue[2]), 120)
        self.assertEqual(rescue[0]["language_model.model.layers.8"], "scale-4-8")
        model = torch.nn.Linear(1,1)
        modules = {name:model for name in canonical_targets()}
        all_modules = {**modules, **{prefix:model for prefix in rescue[0]}}
        with patch("qvla.run_eval_official_quant.validate_targets", return_value=modules), \
             patch.object(model, "named_modules", return_value=iter(all_modules.items())), \
             patch("qvla.run_eval_official_quant.apply_awq_entry") as vision_apply, \
             patch("qvla.awq_block.apply_llama_entry") as language_apply, \
             patch("qvla.awq_block.apply_block_scales") as scales:
            apply_quantization(model, entries, metadata, "awq", {}, 2, 16, awq_plan=rescue, awq_scope="language")
            self.assertEqual(Counter(call.kwargs["bits"] for call in language_apply.call_args_list), {2:168,4:56})
            self.assertEqual(scales.call_count, 32)
            vision_apply.assert_not_called()


if __name__ == "__main__":
    unittest.main()
