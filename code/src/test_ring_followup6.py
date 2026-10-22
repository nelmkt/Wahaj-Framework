"""Tests for paired middle-ring change and prepared subring design."""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_ring_reduced_v8_followup6 import OUT_COMPARISONS
from analyze_pixel_subrings_v8_followup6 import load_new, subring_design
from fw_config import Config
from write_ring_followup6 import TARGET


class RingFollowup6Tests(unittest.TestCase):
    def test_subrings_nest_and_coarse_model_merges_inner_outer(self):
        designs = subring_design(np.array([1., 2.]), np.array([1., 0.]),
                                 np.array([3., 2.]), np.array([5., 4.]),
                                 np.array([9., 8.]), np.array([12., 10.]))
        np.testing.assert_array_equal(designs["fine_three_inner_rings"],
                                      [[1., 1., 2., 2., 4., 3.],
                                       [2., 0., 2., 2., 4., 2.]])
        np.testing.assert_array_equal(designs["coarse_two_inner_rings"],
                                      [[1., 1., 4., 4., 3.],
                                       [2., 0., 4., 4., 2.]])
        with self.assertRaises(ValueError):
            subring_design([1], [1], [0], [2], [3], [4])

    def test_no_unrun_export_is_treated_as_a_result(self):
        with self.assertRaisesRegex(SystemExit, "CSV/JSON are missing"):
            load_new(Path("this-subring-export-does-not-exist.csv"), Config())

    def test_saved_middle_change_is_paired_and_reported(self):
        rows = pd.read_csv(OUT_COMPARISONS)
        row = rows.loc[rows.model.eq("drop_180_300m") &
                       rows.term.eq("external_90_180m")]
        self.assertEqual(len(row), 1)
        r = row.iloc[0]
        self.assertEqual(int(r.valid_paired_block_draws), 2000)
        self.assertLess(float(r.difference_hi_C), 0.)
        self.assertAlmostEqual(float(r.reduced_beta_C_per_pixel) -
                               float(r.full_beta_C_per_pixel),
                               float(r.reduced_minus_full_C_per_pixel))
        if not TARGET.exists():
            self.skipTest("Versioned draft manuscript is not in the public release")
        text = TARGET.read_text(encoding="utf-8")
        self.assertIn("specification sensitive", text)
        self.assertIn("interval excludes zero", text)


if __name__ == "__main__":
    unittest.main()
