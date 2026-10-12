import unittest
import numpy as np

from src.ml_core import (
    confusion_matrix,
    fit_least_squares_svd,
    knn_regression_1d,
    mse,
    polynomial_features_1d,
    radius_regression_1d,
    softmax,
)


class CoreAlgorithmTests(unittest.TestCase):
    def test_mse(self):
        self.assertAlmostEqual(mse(np.array([1, 2]), np.array([1, 4])), 2.0)

    def test_radius_regression(self):
        pred, fallback = radius_regression_1d([0, 1, 2], [0, 10, 20], [1.0, 10.0], 0.25)
        self.assertAlmostEqual(pred[0], 10.0)
        self.assertAlmostEqual(pred[1], 10.0)  # fallback mean
        self.assertAlmostEqual(fallback, 0.5)

    def test_knn_regression(self):
        pred = knn_regression_1d([0, 1, 2, 3], [0, 10, 20, 30], [1.2], 2)
        self.assertAlmostEqual(pred[0], 15.0)

    def test_polynomial_svd_fit(self):
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        y = 1 + 2 * x + 3 * x**2
        X = polynomial_features_1d(x, 2)
        w = fit_least_squares_svd(X, y)
        np.testing.assert_allclose(w, [1, 2, 3], atol=1e-10)

    def test_softmax_rows_sum_to_one(self):
        probs = softmax(np.array([[1000.0, 1001.0], [-1000.0, -999.0]]))
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0])

    def test_confusion_matrix(self):
        cm = confusion_matrix([0, 1, 1], [0, 0, 1], 2)
        np.testing.assert_array_equal(cm, [[1, 0], [1, 1]])


if __name__ == "__main__":
    unittest.main()
