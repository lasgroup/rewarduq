import unittest

import numpy as np

from rewarduq.metrics import compute_default_metrics


class MetricsTester(unittest.TestCase):
    def setUp(self):
        pass

    def assertMetricsAlmostEqual(self, dict1, dict2, rtol=1e-7, atol=1e-8):
        self.assertEqual(set(dict1.keys()), set(dict2.keys()))
        for key in dict1.keys():
            val1, val2 = dict1[key], dict2[key]
            if isinstance(val1, (np.ndarray, np.generic)) or isinstance(val2, (np.ndarray, np.generic)):
                val1_arr, val2_arr = np.asarray(val1), np.asarray(val2)
                self.assertEqual(val1_arr.shape, val2_arr.shape)
                self.assertFalse(np.isnan(val1_arr).any(), f"NaN found in val1 for key '{key}'")
                self.assertFalse(np.isnan(val2_arr).any(), f"NaN found in val2 for key '{key}'")
                self.assertTrue(
                    np.allclose(val1_arr, val2_arr, rtol=rtol, atol=atol),
                    f"Arrays not almost equal for key '{key}'",
                )
            else:
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    self.assertFalse(np.isnan(val1), f"NaN found in val1 for key '{key}'")
                    self.assertFalse(np.isnan(val2), f"NaN found in val2 for key '{key}'")
                    self.assertTrue(
                        np.isclose(val1, val2, rtol=rtol, atol=atol),
                        f"Values not almost equal for key '{key}'",
                    )
                else:
                    self.assertEqual(val1, val2)

    def test_weighting(self):
        sample1 = [[1.0, 0.5, 1.5], [0.0, -0.2, 0.2]]
        sample2 = [[0.5, 0.0, 1.0], [0.1, -0.1, 0.1]]
        sample3 = [[0.0, -0.5, 0.5], [-1.0, -1.5, -0.5]]

        # test that weighting a single sample is equivalent to repeating it

        metrics1a = compute_default_metrics(
            {
                "rewards": np.array([sample1]),
            },
            report_to="none",
        )
        metrics1b = compute_default_metrics(
            {
                "rewards": np.array([sample1, sample1, sample1]),
            },
            report_to="none",
        )
        metrics1c = compute_default_metrics(
            {
                "rewards": np.array([sample1]),
                "weights": np.array([10.0]),
            },
            report_to="none",
        )
        self.assertMetricsAlmostEqual(metrics1a, metrics1b)
        self.assertMetricsAlmostEqual(metrics1a, metrics1c)

        # test that weighting different samples is equivalent to repeating them

        metrics2a = compute_default_metrics(
            {
                "rewards": np.array([sample1, sample1, sample1, sample2, sample2, sample3]),
            },
            report_to="none",
        )
        metrics2b = compute_default_metrics(
            {
                "rewards": np.array([sample1, sample2, sample3]),
                "weights": np.array([3.0, 2.0, 1.0]),
            },
            report_to="none",
        )
        self.assertMetricsAlmostEqual(metrics2a, metrics2b)


if __name__ == "__main__":
    unittest.main()
