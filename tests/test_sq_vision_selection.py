import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "diagnostics"))
from smoothing_selection import excluded_norm

class VisionSelectionTest(unittest.TestCase):
    def test_select_complete_branch_groups_and_exclude_language(self):
        primary = "vision_backbone.featurizer.blocks.0.norm1"
        fused = "vision_backbone.fused_featurizer.blocks.0.norm1"
        language = "language_model.model.layers.0.input_layernorm"
        for selection, kept, omitted in (("only-primary-vision", primary, fused),
                                        ("only-fused-vision", fused, primary)):
            self.assertFalse(excluded_norm(kept, selection))
            self.assertTrue(excluded_norm(omitted, selection))
            self.assertTrue(excluded_norm(language, selection))

if __name__ == "__main__":
    unittest.main()
