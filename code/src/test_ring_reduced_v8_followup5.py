"""Synthetic and saved-output tests for the versioned reduced-ring follow-up."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_ring_reduced_v8_followup5 import (
    OUT_COEFFICIENTS, OUT_COMPARISONS, model_design,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gee"))
from export_external_pixel_subrings_v8_followup5 import (  # noqa: E402
    RADII_M, offsets_for_radii, unique_external_counts,
)


class ReducedRingTests(unittest.TestCase):
    def test_reduced_model_columns_partition_external_pixels(self):
        x = model_design(np.array([1., 2.]), np.array([2., 1.]),
                         np.array([4., 3.]), np.array([7., 8.]))
        np.testing.assert_array_equal(x["full_four_terms"],
                                      [[1., 2., 2., 3.], [2., 1., 2., 5.]])
        np.testing.assert_array_equal(x["drop_180_300m"], x["full_four_terms"][:, :3])
        np.testing.assert_array_equal(x["merge_90_300m"][:, 2], [5., 7.])
        with self.assertRaises(ValueError):
            model_design(np.ones(1), np.array([3.]), np.array([2.]), np.array([4.]))

    def test_unique_subrings_count_external_pixel_once(self):
        own = [(0, 0), (0, 1)]
        pix = {own[0]: "cell", own[1]: "cell",
               (1, 1): "other", (2, 0): "other", (3, 0): "other"}
        offsets = offsets_for_radii((30., 0., 0., 0., 30., 0.), RADII_M)
        counts = unique_external_counts(own, "cell", pix, offsets, RADII_M)
        self.assertEqual(counts[30], 1)
        self.assertEqual(counts[60], 2)
        self.assertEqual(counts[90], 3)
        self.assertEqual(counts[30] + (counts[60]-counts[30]) +
                         (counts[90]-counts[60]), counts[90])

    def test_saved_reduced_models_reconcile(self):
        coef = pd.read_csv(OUT_COEFFICIENTS)
        compare = pd.read_csv(OUT_COMPARISONS)
        self.assertEqual((len(coef), len(compare)), (10, 4))
        self.assertTrue(coef.design_rank.eq(coef.model.map(
            {"full_four_terms": 4, "drop_180_300m": 3, "merge_90_300m": 3})).all())
        self.assertTrue(compare.valid_paired_block_draws.eq(2000).all())
        full = coef.loc[coef.model.eq("full_four_terms")].set_index("term")
        for row in compare.itertuples():
            self.assertAlmostEqual(row.full_beta_C_per_pixel,
                                   full.loc[row.term, "beta_C_per_pixel"])
            self.assertAlmostEqual(row.reduced_beta_C_per_pixel - row.full_beta_C_per_pixel,
                                   row.reduced_minus_full_C_per_pixel)


if __name__ == "__main__":
    unittest.main()
