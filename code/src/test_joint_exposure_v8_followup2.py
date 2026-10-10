"""Synthetic tests for new joint-exposure and expanded-balance arithmetic."""
import unittest

import numpy as np

from analyze_isolation_reweighting_v8 import entropy_weights
from analyze_joint_exposure_v8_followup2 import (
    OUT_MANUSCRIPT, calibration_matrix, joint_fit, relative_spread,
)


class JointExposureFollowupTests(unittest.TestCase):
    def test_zero_intercept_joint_fit_recovers_two_known_coefficients(self):
        own = np.array([1., 2., 3., 4., 5.])
        external = np.array([0., 2., 1., 5., 3.])
        delta = -1.25 * own - 0.20 * external
        beta = joint_fit(own, external, delta, np.array([1., 2., 3., 1., 4.]))
        np.testing.assert_allclose(beta, [-1.25, -0.20], atol=1e-12)

    def test_joint_fit_rejects_collinear_exposures(self):
        own = np.array([1., 2., 3.])
        beta = joint_fit(own, 2 * own, -own, np.ones(3))
        self.assertTrue(np.isnan(beta).all())

    def test_expanded_calibration_balances_belt_and_pixel_mix(self):
        combos = np.array(np.meshgrid([-1., 0., 1.], [-1., 0., 1.],
                                      [-1., 0., 1.], [0., 1.], [0., 1.])).reshape(5, -1).T
        design = calibration_matrix(combos[:, :3], combos[:, 3], combos[:, 4])
        target_weight = np.exp(design @ np.array([.08, -.06, .12, .02, -.01,
                                                  .03, .02, -.03, .01, .35, -.25]))
        target_weight /= target_weight.sum()
        fitted = entropy_weights(design, design, np.ones(len(design)), target_weight)
        self.assertIsNotNone(fitted)
        np.testing.assert_allclose(fitted @ design, target_weight @ design, atol=1e-8)
        self.assertAlmostEqual(fitted.sum(), 1.)

    def test_relative_spread_uses_absolute_mean(self):
        self.assertAlmostEqual(relative_spread(np.array([-1., -2., -3., -4.])), 1.2)
        self.assertTrue(np.isnan(relative_spread(np.array([-1., 1.]))))

    def test_versioned_manuscript_does_not_claim_contamination_is_identified(self):
        text = OUT_MANUSCRIPT.read_text(encoding="utf-8")
        self.assertIn("does not distinguish neighbour contamination", text)
        self.assertIn("gap persisted at 90 and 180 m", text)
        self.assertIn("combined_class_relative_spread_v8_followup2.csv", text)
        self.assertNotIn("lacks unique external 30 m greened-pixel counts", text)


if __name__ == "__main__":
    unittest.main()
