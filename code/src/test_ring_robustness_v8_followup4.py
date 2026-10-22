"""Synthetic and saved-output checks for ring robustness calculations."""
import unittest

import numpy as np
import pandas as pd

from analyze_ring_robustness_v8_followup4 import (
    OUT_LOO, OUT_PAIR, OUT_SUMMARY, cosine, normalized_condition,
)
from write_ring_robustness_v8_followup4 import TARGET


class RingRobustnessTests(unittest.TestCase):
    def test_cosine_uses_uncentred_vectors(self):
        self.assertAlmostEqual(cosine(np.array([1., 0.]), np.array([0., 1.])), 0.)
        self.assertAlmostEqual(cosine(np.array([2., 0.]), np.array([5., 0.])), 1.)
        self.assertTrue(np.isnan(cosine(np.zeros(2), np.ones(2))))

    def test_normalized_condition_removes_column_units(self):
        x = np.array([[1., 0.], [0., 100.]])
        self.assertAlmostEqual(normalized_condition(x), 1.)

    def test_saved_block_deletions_reconcile_with_summary(self):
        loo = pd.read_csv(OUT_LOO)
        summary = pd.read_csv(OUT_SUMMARY).iloc[0]
        pair = pd.read_csv(OUT_PAIR)
        self.assertEqual((len(loo), len(pair)), (32, 12))
        self.assertEqual(loo.deleted_block.nunique(), 32)
        self.assertTrue((loo.design_rank == 4).all())
        self.assertEqual(int(summary.sign_reversals),
                         int((loo.beta_external_0_90m_C_per_pixel >= 0).sum()))
        self.assertAlmostEqual(summary.minimum_leave_one_out_beta_C_per_pixel,
                               loo.beta_external_0_90m_C_per_pixel.min())
        self.assertAlmostEqual(summary.maximum_leave_one_out_beta_C_per_pixel,
                               loo.beta_external_0_90m_C_per_pixel.max())
        self.assertAlmostEqual(summary.maximum_absolute_change_C_per_pixel,
                               loo.change_external_0_90m_C_per_pixel.abs().max())
        self.assertTrue((pair.external_three_rank == 3).all())

    def test_manuscript_keeps_distance_limit_and_tension(self):
        if not TARGET.exists():
            self.skipTest("Versioned draft manuscript is not in the public release")
        text = TARGET.read_text(encoding="utf-8")
        self.assertIn("Neither supports a claim of zero association beyond 90 m", text)
        self.assertIn("no unique counts at 30 or 60 m", text)
        self.assertIn("one-block leverage", text)


if __name__ == "__main__":
    unittest.main()
