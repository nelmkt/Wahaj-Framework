"""Synthetic and saved-output checks for the versioned ring follow-up."""
import unittest

import numpy as np
import pandas as pd

from analyze_exposure_rings_v8_followup3 import (
    OUT_COLLIN, OUT_RING, OUT_SUBGROUP, two_column_diagnostics, wls_no_intercept,
)
from write_exposure_manuscript_v8_followup3 import SUPPLEMENT, TARGET


class ExposureRingTests(unittest.TestCase):
    def test_four_nonoverlapping_columns_recover_known_coefficients(self):
        x = np.array([[1., 1., 0., 2.], [2., 0., 2., 1.],
                      [1., 2., 3., 0.], [2., 4., 1., 3.],
                      [1., 0., 1., 4.], [2., 3., 2., 2.]])
        beta = np.array([-1., -.3, -.2, -.1])
        np.testing.assert_allclose(wls_no_intercept(x, x @ beta,
                                                     np.array([1., 2., 3., 1., 1., 2.])), beta, atol=1e-12)

    def test_rank_deficiency_is_not_filled_with_coefficients(self):
        own = np.arange(1., 7.)
        x = np.column_stack((own, own, own * 2, own * 3))
        self.assertTrue(np.isnan(wls_no_intercept(x, -own, np.ones(len(own)))).all())

    def test_uncentred_cosine_and_zero_share(self):
        values = two_column_diagnostics(np.array([1., 1., 1.]), np.array([0., 1., 0.]))
        self.assertAlmostEqual(values["uncentred_cosine"], 1 / np.sqrt(3))
        self.assertAlmostEqual(values["zero_external_share"], 2/3)
        self.assertEqual(values["algebraic_rank"], 2)

    def test_versioned_outputs_reconcile_and_flag_small_groups(self):
        ring = pd.read_csv(OUT_RING)
        col = pd.read_csv(OUT_COLLIN)
        subgroup = pd.read_csv(OUT_SUBGROUP)
        self.assertEqual((len(ring), len(col), len(subgroup)), (16, 15, 12))
        self.assertTrue((ring.algebraic_rank == 4).all())
        self.assertTrue((col.algebraic_rank == 2).all())
        self.assertTrue((subgroup.loc[subgroup.isolated_cells < 20, "status"] ==
                         "diagnostic_only_under_20_cells").all())
        self.assertTrue(subgroup.loc[subgroup.status.eq("diagnostic_only_under_20_cells"),
                                     "difference_lo_C"].isna().all())
        self.assertEqual(subgroup.loc[subgroup.subgroup.eq("exactly two"),
                                      "isolated_cells"].tolist(), [13, 7, 4])

    def test_main_and_supplement_are_separated(self):
        if not TARGET.exists() or not SUPPLEMENT.exists():
            self.skipTest("Versioned draft manuscript is not in the public release")
        main = TARGET.read_text(encoding="utf-8")
        supplement = SUPPLEMENT.read_text(encoding="utf-8")
        self.assertIn("The point gaps attenuate with radius", main)
        self.assertIn("SUPPLEMENT_RELATIVE_SPREAD_V8_FOLLOWUP3.md", main)
        self.assertNotIn("range divided by the absolute four-class mean", main)
        self.assertIn("0.628 [0.556, 0.754]", supplement)


if __name__ == "__main__":
    unittest.main()
