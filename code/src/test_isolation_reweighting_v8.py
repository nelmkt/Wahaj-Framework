"""Small synthetic checks for the saved-panel reweighting arithmetic."""

import unittest

import numpy as np

from analyze_isolation_reweighting_v8 import entropy_weights, interval, moments, slope


class IsolationReweightingTests(unittest.TestCase):
    def test_entropy_weights_match_joint_moments_without_duplicate_collapse(self):
        grid = np.array(np.meshgrid([-1.0, 0.0, 1.0],
                                    [-1.0, 0.0, 1.0],
                                    [-1.0, 0.0, 1.0])).reshape(3, -1).T
        design = moments(grid, np.zeros(3), np.ones(3))
        tilt = np.array([0.25, -0.15, 0.35, 0.08, -0.04,
                         0.06, 0.1, -0.07, 0.03])
        target_weight = np.exp(design @ tilt)
        target_weight /= target_weight.sum()
        fitted = entropy_weights(design, design, np.ones(len(grid)), target_weight)
        self.assertIsNotNone(fitted)
        np.testing.assert_allclose(fitted @ design,
                                   target_weight @ design, atol=1e-8)
        np.testing.assert_allclose(fitted, target_weight, atol=1e-8)
        self.assertAlmostEqual(fitted.sum(), 1.0)
        self.assertGreater(1 / np.sum(fitted ** 2), 1)

    def test_missing_source_support_returns_none(self):
        source = np.zeros((2, 9))
        target = np.ones((1, 9))
        self.assertIsNone(entropy_weights(source, target,
                                          np.ones(2), np.ones(1)))

    def test_zero_intercept_slope_and_finite_interval(self):
        dose = np.array([1.0, 2.0, 3.0])
        effect = -2.0 * dose
        self.assertAlmostEqual(slope(dose, effect, np.array([1.0, 4.0, 2.0])), -2.0)
        lo, hi, n = interval([1.0, np.nan, 3.0])
        self.assertEqual(n, 2)
        self.assertGreater(lo, 1.0)
        self.assertLess(hi, 3.0)


if __name__ == "__main__":
    unittest.main()
