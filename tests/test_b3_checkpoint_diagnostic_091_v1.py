"""CPU mathematical contracts; no model/GPU or held-out labels are loaded."""
import unittest

import numpy as np

from scripts.diagnose_b3_checkpoints_091_v1 import action_metrics, low_rank_norms, gripper_metrics


class CheckpointDiagnosticContracts(unittest.TestCase):
    def test_rank_gram_norm_matches_explicit_nonsquare_product(self):
        # The runtime uses rank-sized Grams to avoid an output x input matrix.
        generator = np.random.default_rng(22)
        a, b = generator.normal(size=(8, 31)), generator.normal(size=(23, 8))
        result = low_rank_norms(a, b)
        self.assertAlmostEqual(result["BA_frobenius"], np.linalg.norm(b @ a), places=11)
        self.assertEqual(low_rank_norms(a, np.zeros_like(b))["BA_frobenius"], 0)

    def test_action_aggregation_preserves_step_and_dimension(self):
        teacher = np.zeros((2, 8, 7))
        prediction = teacher.copy()
        prediction[0, 3, 6] = 2
        result = action_metrics(prediction, teacher)
        self.assertAlmostEqual(result["mse"], 4 / 112)
        self.assertAlmostEqual(result["mse_per_dim"][6], 4 / 16)
        self.assertAlmostEqual(result["mse_per_chunk_step"][3], 4 / 14)
        self.assertEqual(result["mse_chunk_by_dim"][3][6], 2)
        self.assertAlmostEqual(result["smooth_l1_beta_0_1"], 1.95 / 112)
        with self.assertRaises(ValueError):
            action_metrics(prediction[0], teacher[0])

    def test_native_gripper_processor_and_exact_zero_tie(self):
        # A supplied deployment processor must be called per action, including
        # its inversion. Margin alone must not invent a binary decision at 0.5.
        calls = []

        def deployment(action, family):
            calls.append(family)
            action[-1] = -np.sign(2 * action[-1] - 1)
            return action

        raw = np.zeros((1, 8, 7))
        raw[0, :, -1] = [0, .5, 1, 0, .5, 1, 0, 1]
        teacher = raw.copy()
        teacher[0, 1, -1] = 1
        result = gripper_metrics(raw, teacher, deployment, "openvla")
        self.assertEqual(len(calls), 16)
        self.assertEqual(result["candidate_decision"][0][:3], [1, 0, -1])
        self.assertEqual(result["candidate_zero_decisions"], 2)
        self.assertEqual(result["disagreements"], 1)
        self.assertEqual(result["decision_disagreement"], 1 / 8)
        self.assertEqual(result["candidate_prebinarization_signed_margin_2raw_minus1"][0][1], 0)


if __name__ == "__main__":
    unittest.main()
