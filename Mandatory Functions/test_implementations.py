"""Deterministic contract, course-reference, and numerical sanity checks."""

import unittest

import numpy as np

import implementations as impl


class MandatoryFunctionsTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(433)
        self.tx = rng.normal(size=(10, 3))
        self.y = rng.normal(size=10)
        self.labels = (self.y > 0).astype(float)
        self.initial_w = np.array([0.1, -0.2, 0.3])

    def test_shapes_final_losses_and_input_preservation(self):
        original_inputs = [a.copy() for a in (self.y, self.labels, self.tx, self.initial_w)]
        calls = [
            (impl.mean_squared_error_gd, self.y, (self.initial_w, 7, 0.1), False),
            (impl.mean_squared_error_sgd, self.y, (self.initial_w, 7, 0.1), False),
            (impl.least_squares, self.y, (), False),
            (impl.ridge_regression, self.y, (0.2,), False),
            (impl.logistic_regression, self.labels, (self.initial_w, 7, 0.1), True),
            (impl.reg_logistic_regression, self.labels, (0.2, self.initial_w, 7, 0.1), True),
        ]
        np.random.seed(433)
        for method, y, args, logistic in calls:
            with self.subTest(method=method.__name__):
                result = method(y, self.tx, *args)
                self.assertIsInstance(result, tuple)
                self.assertEqual(len(result), 2)
                w, loss = result
                self.assertEqual(w.shape, (3,))
                self.assertTrue(np.isscalar(loss))
                self.assertTrue(np.isfinite(loss))
                if logistic:
                    probability = 1 / (1 + np.exp(-self.tx @ w))
                    expected = -np.mean(y * np.log(probability) + (1 - y) * np.log1p(-probability))
                else:
                    expected = 0.5 * np.mean((y - self.tx @ w) ** 2)
                self.assertAlmostEqual(loss, expected)
        for actual, original in zip(
            (self.y, self.labels, self.tx, self.initial_w), original_inputs
        ):
            np.testing.assert_array_equal(actual, original)

    def test_zero_iterations(self):
        for method, y, extra in [
            (impl.mean_squared_error_gd, self.y, ()),
            (impl.mean_squared_error_sgd, self.y, ()),
            (impl.logistic_regression, self.labels, ()),
            (impl.reg_logistic_regression, self.labels, (0.2,)),
        ]:
            with self.subTest(method=method.__name__):
                w, loss = method(y, self.tx, *extra, self.initial_w, 0, 0.1)
                np.testing.assert_array_equal(w, self.initial_w)
                if method in (impl.mean_squared_error_gd, impl.mean_squared_error_sgd):
                    expected = 0.5 * np.mean((y - self.tx @ self.initial_w) ** 2)
                else:
                    scores = self.tx @ self.initial_w
                    expected = np.mean(np.log1p(np.exp(scores)) - y * scores)
                self.assertAlmostEqual(loss, expected)

    def test_gd_exact_update_count(self):
        # One sample with x=1, y=1 yields w_k = 1 - (1-gamma)^k.
        w, loss = impl.mean_squared_error_gd(
            np.ones(1), np.ones((1, 1)), np.zeros(1), 5, 0.2
        )
        np.testing.assert_allclose(w, [1 - 0.8**5])
        self.assertAlmostEqual(loss, 0.5 * 0.8**10)

    def test_sgd_lab_sampling_and_update_count(self):
        # Replay lab 2's batch_iter(batch_size=1, num_batches=1) sampling.
        expected = self.initial_w.copy()
        np.random.seed(433)
        for _ in range(17):
            index = np.random.randint(len(self.y), size=1)[0]
            residual = self.y[index] - self.tx[index] @ expected
            expected = expected + 0.1 * self.tx[index] * residual
        expected_next_draw = np.random.randint(1000)
        np.random.seed(433)
        w, _ = impl.mean_squared_error_sgd(self.y, self.tx, self.initial_w, 17, 0.1)
        np.testing.assert_allclose(w, expected)
        self.assertEqual(np.random.randint(1000), expected_next_draw)

    def test_least_squares_normal_equations(self):
        w, _ = impl.least_squares(self.y, self.tx)
        np.testing.assert_allclose(self.tx.T @ (self.tx @ w - self.y), 0, atol=1e-12)

    def test_ridge_lab_reference_and_zero_penalty(self):
        # Reference from labs/ex04/solution/ridge_regression.py.
        w, _ = impl.ridge_regression(
            np.array([0.1, 0.2]), np.array([[2.3, 3.2], [1.0, 0.1]]), 1
        )
        np.testing.assert_allclose(w, [0.03947092, 0.00319628], atol=5e-9)
        np.testing.assert_allclose(
            impl.ridge_regression(self.y, self.tx, 0)[0],
            impl.least_squares(self.y, self.tx)[0],
        )

    def test_logistic_lab_update_references(self):
        # Reference updates from labs/ex05/solution/ex05.ipynb, flattened to 1-D.
        y = np.array([0.0, 1.0])
        tx = np.arange(6).reshape(2, 3)
        initial_w = np.array([0.1, 0.2, 0.3])
        for method, extra, expected in [
            (impl.logistic_regression, (), [0.11037076, 0.17932896, 0.24828716]),
            (impl.reg_logistic_regression, (0.1,), [0.10837076, 0.17532896, 0.24228716]),
        ]:
            with self.subTest(method=method.__name__):
                w, _ = method(y, tx, *extra, initial_w, 1, 0.1)
                np.testing.assert_allclose(w, expected, atol=5e-9)

    def test_logistic_zero_penalty(self):
        ordinary = impl.logistic_regression(self.labels, self.tx, self.initial_w, 9, 0.1)
        regularized = impl.reg_logistic_regression(self.labels, self.tx, 0, self.initial_w, 9, 0.1)
        np.testing.assert_allclose(ordinary[0], regularized[0])
        self.assertEqual(ordinary[1], regularized[1])

    def test_logistic_exact_update_count(self):
        expected = self.initial_w.copy()
        for _ in range(9):
            probability = 1 / (1 + np.exp(-self.tx @ expected))
            expected -= 0.1 * (self.tx.T @ (probability - self.labels) / 10 + 0.4 * expected)
        w, _ = impl.reg_logistic_regression(
            self.labels, self.tx, 0.2, self.initial_w, 9, 0.1
        )
        np.testing.assert_allclose(w, expected)

    def test_logistic_extreme_scores(self):
        scores = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            probabilities = impl._sigmoid(scores)
            self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
            for method, extra in [
                (impl.logistic_regression, ()),
                (impl.reg_logistic_regression, (0.1,)),
            ]:
                _, loss = method(
                    np.array([1.0, 0.0]), np.array([[-1000.0], [1000.0]]),
                    *extra, np.ones(1), 0, 0.1
                )
                self.assertEqual(loss, 1000.0)


if __name__ == '__main__':
    unittest.main()
