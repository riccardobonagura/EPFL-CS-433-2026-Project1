"""Course equivalence and deterministic checks of the experimental toolbox."""

import unittest

import numpy as np

import implementations as mandatory
import implementations_augmented as augmented


class AugmentedImplementationsTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(433)
        self.tx = np.column_stack([np.ones(24), rng.normal(size=(24, 3))])
        self.y = rng.normal(size=24)
        self.labels = (self.y > 0).astype(float)
        self.w = np.array([0.1, -0.2, 0.3, -0.1])

    def assert_result(self, result, expected_w, expected_loss):
        self.assertIsInstance(result, augmented.FitResult)
        self.assertEqual(result.w.shape, expected_w.shape)
        self.assertIsInstance(result.loss, float)
        np.testing.assert_allclose(result.w, expected_w, rtol=1e-12, atol=1e-13)
        self.assertAlmostEqual(result.loss, expected_loss, places=13)

    def test_mandatory_equivalence_for_full_batch_iterative_methods(self):
        pairs = [
            (augmented.fit_mse_gd, mandatory.mean_squared_error_gd, self.y, (), {}),
            (augmented.fit_logistic_regression, mandatory.logistic_regression, self.labels, (), {}),
            (augmented.fit_reg_logistic_regression, mandatory.reg_logistic_regression, self.labels,
             (0.1,), {'penalize_intercept': True}),
        ]
        for fit, reference, y, extra, options in pairs:
            for budget in (0, 1, 17):
                with self.subTest(method=fit.__name__, budget=budget):
                    expected_w, expected_loss = reference(y, self.tx, *extra, self.w, budget, 0.1)
                    result = fit(y, self.tx, *extra, self.w, budget, 0.1, **options)
                    self.assert_result(result, expected_w, expected_loss)
                    self.assertEqual(result.n_updates, budget)
                    self.assertEqual(result.stop_reason, 'max_iters')
                    self.assertFalse(result.converged)
                    self.assertIsNone(result.history)
                    self.assertIsNone(result.best_iteration)
                    self.assertIsNone(result.best_validation_loss)

    def test_mandatory_equivalence_for_closed_form_methods(self):
        expected_w, expected_loss = mandatory.least_squares(self.y, self.tx)
        result = augmented.fit_least_squares(self.y, self.tx, solver='solve')
        self.assert_result(result, expected_w, expected_loss)
        self.assertEqual(result.n_updates, 0)
        self.assertTrue(result.converged)
        self.assertEqual(result.stop_reason, 'closed_form')
        for penalty in (0, 0.1, 1):
            expected_w, expected_loss = mandatory.ridge_regression(self.y, self.tx, penalty)
            result = augmented.fit_ridge_regression(self.y, self.tx, penalty, penalize_intercept=True)
            self.assert_result(result, expected_w, expected_loss)

    def test_newton_reaches_the_long_run_gradient_descent_optimum(self):
        for penalize in (False, True):
            with self.subTest(penalize_intercept=penalize):
                newton = augmented.fit_reg_logistic_newton(self.labels, self.tx, 0.1, np.zeros(4), 50,
                                                           gradient_tol=1e-12, penalize_intercept=penalize)
                descent = augmented.fit_reg_logistic_regression(self.labels, self.tx, 0.1, np.zeros(4), 20000, 0.5,
                                                                penalize_intercept=penalize)
                np.testing.assert_allclose(newton.w, descent.w, atol=1e-10)
                self.assertEqual(newton.stop_reason, 'tolerance')
                self.assertTrue(newton.converged)
                self.assertLess(newton.n_updates, 20)
        expected_w, expected_loss = mandatory.reg_logistic_regression(self.labels, self.tx, 0.1, np.zeros(4), 20000, 0.5)
        result = augmented.fit_reg_logistic_newton(self.labels, self.tx, 0.1, np.zeros(4), 50,
                                                   gradient_tol=1e-12, penalize_intercept=True)
        np.testing.assert_allclose(result.w, expected_w, atol=1e-10)
        self.assertAlmostEqual(result.loss, expected_loss, places=10)

    def test_newton_budget_chunks_validation_and_history(self):
        start = augmented.fit_reg_logistic_newton(self.labels, self.tx, 0.1, self.w, 0)
        np.testing.assert_array_equal(start.w, self.w)
        self.assertEqual((start.n_updates, start.stop_reason, start.converged), (0, 'max_iters', False))
        whole = augmented.fit_reg_logistic_newton(self.labels, self.tx, 0.1, np.zeros(4), 50, gradient_tol=1e-12)
        chunked = augmented.fit_reg_logistic_newton(self.labels, self.tx, 0.1, np.zeros(4), 50,
                                                    gradient_tol=1e-12, chunk_size=5)
        np.testing.assert_allclose(chunked.w, whole.w, atol=1e-14)
        monitored = augmented.fit_reg_logistic_newton(self.labels, self.tx, 0.1, np.zeros(4), 50, gradient_tol=1e-12,
                                                      tx_val=self.tx[:6], y_val=self.labels[:6], return_history=True)
        np.testing.assert_array_equal(monitored.w, whole.w)
        self.assertEqual(monitored.history['update'], list(range(monitored.n_updates + 1)))
        self.assertAlmostEqual(monitored.best_validation_loss, monitored.history['validation_loss'][-1])
        self.assertLessEqual(monitored.history['gradient_norm'][-1], 1e-12)

    def test_full_batch_sizes_match_full_batch_trainers(self):
        for size in (None, len(self.y), len(self.y) + 7):
            with self.subTest(batch_size=size):
                result = augmented.fit_mse_sgd(self.y, self.tx, self.w, 9, 0.1, batch_size=size)
                expected_w, expected_loss = mandatory.mean_squared_error_gd(self.y, self.tx, self.w, 9, 0.1)
                self.assert_result(result, expected_w, expected_loss)

    def test_epoch_traversal_includes_tail_and_counts_updates(self):
        tx = np.arange(1, 6, dtype=float).reshape(-1, 1)
        y = np.arange(5, dtype=float)
        expected = np.zeros(1)
        for indices in ([0, 1], [2, 3], [4], [0, 1]):
            error = y[indices] - tx[indices] @ expected
            expected += 0.01 * (tx[indices].T @ error) / len(indices)
        result = augmented.fit_mse_sgd(y, tx, np.zeros(1), 4, 0.01, batch_size=2, shuffle=False)
        np.testing.assert_allclose(result.w, expected)
        self.assertEqual(result.n_updates, 4)

    def test_mini_batch_rng_is_local_reproducible_and_seed_sensitive(self):
        options = dict(batch_size=4, return_history=True, eval_every=2)
        np.random.seed(12)
        state = np.random.get_state()
        first = augmented.fit_logistic_regression(self.labels, self.tx, self.w, 11, 0.1, seed=42, **options)
        actual_draw = np.random.rand()
        np.random.set_state(state)
        self.assertEqual(actual_draw, np.random.rand())
        second = augmented.fit_logistic_regression(self.labels, self.tx, self.w, 11, 0.1, seed=42, **options)
        different = augmented.fit_logistic_regression(self.labels, self.tx, self.w, 11, 0.1, seed=43, **options)
        np.testing.assert_array_equal(first.w, second.w)
        self.assertFalse(np.allclose(first.w, different.w))
        for key in first.history:
            np.testing.assert_array_equal(first.history[key], second.history[key])

    def test_weighted_losses_and_gradients_match_formulas_and_finite_differences(self):
        weights = np.arange(len(self.y), dtype=float)
        for y, loss_function, gradient_function, logistic in [
            (self.y, augmented._compute_mse, augmented._compute_mse_gradient, False),
            (self.labels, augmented._compute_logistic_loss, augmented._compute_logistic_gradient, True),
        ]:
            with self.subTest(logistic=logistic):
                scores = self.tx @ self.w
                if logistic:
                    losses = np.log1p(np.exp(scores)) - y * scores
                    residual = 1 / (1 + np.exp(-scores)) - y
                else:
                    losses = 0.5 * (y - scores)**2
                    residual = scores - y
                self.assertAlmostEqual(loss_function(y, self.tx, self.w, weights), weights @ losses / weights.sum())
                expected = self.tx.T @ (weights * residual) / weights.sum()
                np.testing.assert_allclose(gradient_function(y, self.tx, self.w, weights), expected)
                numerical = np.empty(len(self.w))
                for index in range(len(self.w)):
                    step = np.zeros(len(self.w))
                    step[index] = 1e-6
                    numerical[index] = (loss_function(y, self.tx, self.w + step, weights)
                                        - loss_function(y, self.tx, self.w - step, weights)) / 2e-6
                np.testing.assert_allclose(expected, numerical, rtol=1e-7, atol=1e-9)

    def test_unit_sample_weights_and_rescaling_invariance(self):
        for fit, y in [(augmented.fit_mse_gd, self.y), (augmented.fit_mse_sgd, self.y),
                       (augmented.fit_logistic_regression, self.labels)]:
            options = {'batch_size': 3} if fit != augmented.fit_mse_gd else {}
            with self.subTest(method=fit.__name__):
                ordinary = fit(y, self.tx, self.w, 13, 0.1, **options)
                unit = fit(y, self.tx, self.w, 13, 0.1, sample_weight=np.ones(len(y)), **options)
                np.testing.assert_allclose(ordinary.w, unit.w, atol=1e-13)
                self.assertAlmostEqual(ordinary.loss, unit.loss)
                weights = np.arange(len(y), dtype=float)
                weighted = fit(y, self.tx, self.w, 13, 0.1, sample_weight=weights, **options)
                scaled = fit(y, self.tx, self.w, 13, 0.1, sample_weight=weights * 1e300, **options)
                np.testing.assert_allclose(weighted.w, scaled.w, atol=1e-13)
                self.assertAlmostEqual(weighted.loss, scaled.loss)

    def test_weighted_single_row_batches_keep_global_weighting_and_allow_zero_batches(self):
        y, tx = np.array([1.0, 2.0]), np.ones((2, 1))
        result = augmented.fit_mse_sgd(y, tx, np.zeros(1), 2, 0.1, shuffle=False,
                                        sample_weight=np.array([0.0, 2.0]))
        # First row has zero data gradient. Second gradient is -4: N * q_i * residual.
        np.testing.assert_allclose(result.w, [0.4])
        self.assertAlmostEqual(result.loss, 0.5 * (2 - 0.4)**2)
        regularized = augmented.fit_reg_logistic_regression(
            np.array([0, 1]), tx, 0.5, np.ones(1), 1, 0.1, batch_size=1,
            shuffle=False, sample_weight=np.array([0.0, 2.0]), penalize_intercept=True,
        )
        np.testing.assert_allclose(regularized.w, [0.9])

    def test_intercept_masks_control_logistic_penalty(self):
        baseline = mandatory._compute_logistic_gradient(self.labels, self.tx, self.w)
        for index in (0, 2):
            mask = np.ones(4)
            mask[index] = 0
            result = augmented.fit_reg_logistic_regression(self.labels, self.tx, 0.2, self.w, 1, 0.1,
                                                           intercept_index=index)
            expected = self.w - 0.1 * (baseline + 0.4 * mask * self.w)
            np.testing.assert_allclose(result.w, expected)
            self.assertAlmostEqual(result.loss, mandatory._compute_logistic_loss(self.labels, self.tx, result.w))

    def test_ridge_exempts_supplied_intercept_and_reports_unpenalized_loss(self):
        tx, y = np.ones((5, 1)), np.full(5, 6.0)
        exempt = augmented.fit_ridge_regression(y, tx, 1)
        penalized = augmented.fit_ridge_regression(y, tx, 1, penalize_intercept=True)
        np.testing.assert_allclose(exempt.w, [6])
        np.testing.assert_allclose(penalized.w, [2])
        self.assertAlmostEqual(penalized.loss, 8)

    def test_early_stopping_restores_best_checkpoint_and_recomputes_training_loss(self):
        y, tx, initial = np.ones(1), np.ones((1, 1)), np.zeros(1)
        result = augmented.fit_mse_gd(y, tx, initial, 20, 0.5, tx_val=tx, y_val=np.array([0.5]),
                                      early_stopping=True, eval_every=1, patience=2, return_history=True)
        self.assertTrue(result.converged)
        self.assertEqual(result.stop_reason, 'early_stopping')
        self.assertEqual(result.n_updates, 3)
        self.assertEqual(result.best_iteration, 1)
        self.assertEqual(result.best_validation_loss, 0)
        np.testing.assert_array_equal(result.w, [0.5])
        self.assertEqual(result.loss, 0.125)
        self.assertNotEqual(result.loss, result.history['train_loss'][-1])

    def test_validation_monitoring_without_early_stopping_returns_last_weights(self):
        result = augmented.fit_mse_gd(np.ones(1), np.ones((1, 1)), np.zeros(1), 3, 0.5,
                                      tx_val=np.ones((1, 1)), y_val=np.array([0.5]), eval_every=1)
        np.testing.assert_array_equal(result.w, [0.875])
        self.assertEqual(result.best_iteration, 1)
        self.assertEqual(result.stop_reason, 'max_iters')
        self.assertFalse(result.converged)

    def test_early_stopping_mode_selects_best_even_when_budget_exhausted(self):
        result = augmented.fit_mse_gd(np.ones(1), np.ones((1, 1)), np.zeros(1), 2, 0.5,
                                      tx_val=np.ones((1, 1)), y_val=np.array([0.5]),
                                      early_stopping=True, patience=10, eval_every=1)
        np.testing.assert_array_equal(result.w, [0.5])
        self.assertEqual(result.n_updates, 2)
        self.assertFalse(result.converged)
        self.assertEqual(result.stop_reason, 'max_iters')

    def test_early_stopping_tolerance_is_strict_and_initial_weights_can_win(self):
        result = augmented.fit_mse_gd(np.ones(1), np.ones((1, 1)), np.zeros(1), 9, 0.5,
                                      tx_val=np.ones((1, 1)), y_val=np.ones(1),
                                      early_stopping=True, patience=1, tol=0.375, eval_every=1)
        # Exactly 0.375 improvement is not greater than tol.
        self.assertEqual(result.n_updates, 1)
        self.assertEqual(result.best_iteration, 0)
        np.testing.assert_array_equal(result.w, [0])

    def test_validation_never_affects_gradients_and_remains_unweighted(self):
        weights = np.arange(1, 25, dtype=float)
        plain = augmented.fit_logistic_regression(self.labels, self.tx, self.w, 11, 0.1, sample_weight=weights)
        monitored = augmented.fit_logistic_regression(self.labels, self.tx, self.w, 11, 0.1,
                                                       sample_weight=weights, tx_val=self.tx,
                                                       y_val=1 - self.labels, return_history=True)
        np.testing.assert_array_equal(plain.w, monitored.w)
        expected_val = augmented._compute_logistic_loss(1 - self.labels, self.tx, monitored.w)
        self.assertEqual(monitored.history['validation_loss'][-1], expected_val)

    def test_gradient_tolerance_uses_full_objective_gradient(self):
        result = augmented.fit_mse_gd(np.ones(1), np.ones((1, 1)), np.zeros(1), 9, 1,
                                      gradient_tol=0, eval_every=1)
        self.assertEqual(result.stop_reason, 'tolerance')
        self.assertEqual(result.n_updates, 1)
        self.assertTrue(result.converged)
        # The first row has zero gradient, but the full gradient is nonzero.
        result = augmented.fit_mse_sgd(np.array([0.0, 1]), np.ones((2, 1)), np.zeros(1), 1, 0.1,
                                       shuffle=False, gradient_tol=0.1, eval_every=1)
        self.assertEqual(result.stop_reason, 'max_iters')
        regularized = augmented.fit_reg_logistic_regression(np.array([0]), np.zeros((1, 1)), 1,
                                                            np.ones(1), 1, 0.1, gradient_tol=0.1,
                                                            penalize_intercept=True, eval_every=1)
        self.assertEqual(regularized.stop_reason, 'max_iters')

    def test_history_checkpoints_are_aligned_and_include_final_budget(self):
        result = augmented.fit_mse_sgd(self.y, self.tx, self.w, 7, 0.1, batch_size=3,
                                       eval_every=3, return_history=True)
        self.assertEqual(result.history['update'], [0, 3, 6, 7])
        self.assertTrue(all(len(values) == 4 for values in result.history.values()))
        self.assertTrue(np.isnan(result.history['validation_loss']).all())
        self.assertNotIn('weights', result.history)
        self.assertEqual(result.history['learning_rate'][0], 0)

    def test_learning_rate_schedules_use_one_based_update_indices(self):
        for schedule, decay in [('constant', 0), ('sqrt', 0), ('inverse', 0.5)]:
            with self.subTest(schedule=schedule):
                result = augmented.fit_mse_gd(np.ones(1), np.ones((1, 1)), np.zeros(1), 3, 0.2,
                                              lr_schedule=schedule, lr_decay=decay,
                                              eval_every=1, return_history=True)
                rates = [0.2 if schedule == 'constant' else 0.2 / np.sqrt(t) if schedule == 'sqrt'
                         else 0.2 / (1 + decay * t) for t in (1, 2, 3)]
                expected = 0.0
                for rate in rates:
                    expected -= rate * (expected - 1)
                np.testing.assert_allclose(result.w, [expected])
                np.testing.assert_allclose(result.history['learning_rate'][1:], rates)

    def test_nonfinite_optimization_is_detected_at_checkpoint(self):
        result = augmented.fit_mse_gd(np.ones(1), np.array([[2.0]]), np.zeros(1), 9, 1e308,
                                      eval_every=2, return_history=True)
        self.assertEqual(result.stop_reason, 'non_finite')
        self.assertFalse(result.converged)
        self.assertEqual(result.n_updates, 2)
        self.assertFalse(np.isfinite(result.loss))

    def test_stable_logistic_extreme_scores(self):
        tx, w, y = np.eye(2), np.array([1000.0, -1000]), np.array([1, 0])
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            probabilities = augmented.logistic_predict_proba(tx, w)
            result = augmented.fit_logistic_regression(y, tx, w, 0, 0.1)
        np.testing.assert_array_equal(probabilities, [1, 0])
        self.assertEqual(result.loss, 0)
        wrong = augmented.fit_logistic_regression(1 - y, tx, w, 0, 0.1)
        self.assertEqual(wrong.loss, 1000)

    def test_explicit_pseudoinverse_handles_rank_deficiency_without_fallback(self):
        tx, y = np.ones((2, 2)), np.array([1.0, 3])
        with self.assertRaises(np.linalg.LinAlgError):
            augmented.fit_least_squares(y, tx)
        result = augmented.fit_least_squares(y, tx, solver='pinv')
        np.testing.assert_allclose(result.w, [1, 1])
        self.assertAlmostEqual(result.loss, 0.5)
        np.testing.assert_allclose(tx.T @ (tx @ result.w - y), 0, atol=1e-12)

    def test_class_weight_helper(self):
        weights = augmented.make_class_weights(np.array([0, 1, 1, 0]), positive_weight=4, negative_weight=0.5)
        np.testing.assert_array_equal(weights, [0.5, 4, 4, 0.5])
        with self.assertRaises(ValueError):
            augmented.make_class_weights(np.array([-1, 1]))
        with self.assertRaises(ValueError):
            augmented.make_class_weights(np.array([0, 1]), positive_weight=0, negative_weight=0)

    def test_prediction_helpers_and_threshold_boundaries(self):
        tx, w = np.array([[-1.0], [0], [1]]), np.ones(1)
        np.testing.assert_array_equal(augmented.linear_predict_scores(tx, w), [-1, 0, 1])
        np.testing.assert_array_equal(augmented.linear_predict(tx, w), [-1, 1, 1])
        np.testing.assert_array_equal(augmented.linear_predict(tx, w, positive_label=9, negative_label=3), [3, 9, 9])
        np.testing.assert_array_equal(augmented.logistic_predict(tx, w), [0, 1, 1])
        np.testing.assert_array_equal(augmented.logistic_predict(tx, w, threshold=np.inf), [0, 0, 0])

    def test_classification_metrics_known_confusion_matrix_and_zero_denominators(self):
        metrics = augmented.classification_metrics(np.array([1, 1, -1, -1, -1]), np.array([1, -1, 1, -1, -1]))
        for key, expected in {'tp': 1, 'tn': 2, 'fp': 1, 'fn': 1, 'accuracy': 0.6, 'precision': 0.5,
                              'recall': 0.5, 'specificity': 2 / 3, 'f1': 0.5,
                              'balanced_accuracy': (0.5 + 2 / 3) / 2}.items():
            self.assertEqual(metrics[key], expected)
        no_positives = augmented.classification_metrics(np.zeros(3), np.zeros(3))
        self.assertEqual(no_positives['precision'], 0)
        self.assertEqual(no_positives['recall'], 0)
        self.assertEqual(no_positives['f1'], 0)
        empty = augmented.classification_metrics(np.array([]), np.array([]))
        self.assertTrue(all(value == 0 for value in empty.values()))
        custom = augmented.classification_metrics(np.array(['yes', 'no']), np.array(['yes', 'no']), positive_label='yes')
        self.assertEqual(custom['accuracy'], 1)

    def test_threshold_selection_known_optimum_and_ties(self):
        scores, y = np.array([0.1, 0.4, 0.6, 0.9]), np.array([0, 0, 1, 1])
        threshold, metrics = augmented.find_best_threshold(y, scores, metric='f1', thresholds=[0.3, 0.5, 0.7])
        self.assertEqual(threshold, 0.5)
        self.assertEqual(metrics['f1'], 1)
        threshold, _ = augmented.find_best_threshold(np.array([0, 1]), np.array([0.1, 0.9]), thresholds=[0.2, 0.5])
        self.assertEqual(threshold, 0.5)
        threshold, _ = augmented.find_best_threshold(np.array([-1, -1, 1, 1]), np.array([-2.0, -1, 1, 2]), thresholds=[0.5, -0.5])
        self.assertEqual(threshold, -0.5)
        threshold, metrics = augmented.find_best_threshold(np.array([0, 1]), np.array([0.1, 0.9]),
                                                            metric='recall', thresholds=[0, 0.5])
        self.assertEqual(threshold, 0.5)
        self.assertEqual(metrics['balanced_accuracy'], 1)

    def test_threshold_cumulative_counts_match_direct_predictions_with_ties(self):
        scores = np.array([-2.0, -1, -1, 0, 0, 1, 1, 2])
        y = np.array([-1, 1, -1, 1, -1, 1, 1, -1])
        candidates = [-np.inf, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, np.inf]
        for metric in ('balanced_accuracy', 'precision', 'recall', 'f1'):
            with self.subTest(metric=metric):
                results = [(threshold, augmented.classification_metrics(y, np.where(scores >= threshold, 1, -1)))
                           for threshold in sorted(candidates)]
                expected = max(results, key=lambda item: (item[1][metric], item[1]['balanced_accuracy'], -abs(item[0])))
                threshold, metrics = augmented.find_best_threshold(y, scores, metric=metric, thresholds=candidates)
                self.assertEqual(threshold, expected[0])
                self.assertEqual(metrics, expected[1])
        _, metrics = augmented.find_best_threshold(y, scores)
        self.assertTrue(np.isfinite(metrics['balanced_accuracy']))

    def test_invalid_shapes_and_inputs_raise_value_error(self):
        for y, tx, w in [(self.y[:, None], self.tx, self.w), (self.y, self.tx.ravel(), self.w),
                         (self.y[:-1], self.tx, self.w), (self.y, self.tx, self.w[:, None]),
                         (self.y, self.tx, self.w[:-1]), (np.array([]), np.empty((0, 4)), self.w)]:
            with self.subTest(shapes=(y.shape, tx.shape, w.shape)), self.assertRaises(ValueError):
                augmented.fit_mse_gd(y, tx, w, 1, 0.1)
        for options in [{'gamma': 0}, {'gamma': np.inf}, {'gamma': np.nan}, {'max_iters': -1},
                        {'max_iters': 1.5}, {'eval_every': 0}, {'patience': 0}, {'tol': -1},
                        {'gradient_tol': -1}, {'lr_decay': -1}, {'lr_schedule': 'adam'},
                        {'tx_val': self.tx}, {'y_val': self.y}, {'early_stopping': True},
                        {'sample_weight': np.ones((24, 1))}, {'sample_weight': -np.ones(24)},
                        {'sample_weight': np.zeros(24)}, {'sample_weight': np.full(24, np.nan)},
                        {'tx_val': self.tx[:, :-1], 'y_val': self.y}]:
            arguments = dict(max_iters=1, gamma=0.1)
            arguments.update(options)
            with self.subTest(options=list(options)), self.assertRaises(ValueError):
                augmented.fit_mse_gd(self.y, self.tx, self.w, **arguments)
        for size in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                augmented.fit_mse_sgd(self.y, self.tx, self.w, 1, 0.1, batch_size=size)
        with self.assertRaises(ValueError):
            augmented.fit_logistic_regression(self.y, self.tx, self.w, 1, 0.1)
        with self.assertRaises(ValueError):
            augmented.fit_logistic_regression(self.labels, self.tx, self.w, 1, 0.1,
                                              tx_val=self.tx, y_val=np.full(24, -1))
        for penalty in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                augmented.fit_ridge_regression(self.y, self.tx, penalty)
        for index in (-1, 4):
            with self.assertRaises(ValueError):
                augmented.fit_reg_logistic_regression(self.labels, self.tx, 0.1, self.w, 1, 0.1,
                                                      intercept_index=index)
        with self.assertRaises(ValueError):
            augmented.fit_least_squares(self.y, self.tx, solver='auto')

    def test_invalid_predictions_metrics_and_threshold_inputs(self):
        for action in [
            lambda: augmented.linear_predict_scores(self.tx, self.w[:, None]),
            lambda: augmented.logistic_predict(self.tx, self.w, threshold=np.nan),
            lambda: augmented.classification_metrics(np.array([0, 1]), np.array([0])),
            lambda: augmented.classification_metrics(np.array([0, 1, 2]), np.array([0, 1, 2])),
            lambda: augmented.find_best_threshold(np.array([0]), np.array([np.nan])),
            lambda: augmented.find_best_threshold(np.array([0]), np.array([0.5]), metric='accuracy'),
            lambda: augmented.find_best_threshold(np.array([0]), np.array([0.5]), thresholds=[]),
        ]:
            with self.assertRaises(ValueError):
                action()

    def test_inputs_are_not_mutated(self):
        weights = np.arange(1, 25, dtype=float)
        originals = [value.copy() for value in (self.labels, self.tx, self.w, weights)]
        augmented.fit_reg_logistic_regression(self.labels, self.tx, 0.1, self.w, 11, 0.1,
                                              batch_size=3, sample_weight=weights,
                                              tx_val=self.tx, y_val=self.labels, early_stopping=True,
                                              return_history=True, eval_every=2)
        for value, original in zip((self.labels, self.tx, self.w, weights), originals):
            np.testing.assert_array_equal(value, original)


if __name__ == '__main__':
    unittest.main()
