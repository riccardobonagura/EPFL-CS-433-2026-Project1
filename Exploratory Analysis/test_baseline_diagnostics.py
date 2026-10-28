"""Checks of sampling, training-only preprocessing, metrics and mandatory calls."""

import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import baseline_diagnostics as diagnostics


class BaselineDiagnosticsTests(unittest.TestCase):
    """Unit and end-to-end checks of baseline_diagnostics.py on synthetic data."""
    def test_stratified_sample_and_split_are_reproducible_and_disjoint(self):
        """Same seed gives the same sample/split; parts are disjoint and stratified."""
        y = np.r_[-np.ones(900), np.ones(100)]
        sample = diagnostics.stratified_subsample(y, 200)
        np.testing.assert_array_equal(sample, diagnostics.stratified_subsample(y, 200))
        self.assertEqual(len(np.unique(sample)), 200)
        self.assertEqual(np.sum(y[sample] == 1), 20)
        train, validation = diagnostics.stratified_split(y[sample])
        self.assertEqual(len(train), 160)
        self.assertEqual(len(validation), 40)
        self.assertFalse(np.intersect1d(train, validation).size)
        np.testing.assert_array_equal(np.sort(np.r_[train, validation]), np.arange(200))
        self.assertEqual(np.sum(y[sample][validation] == 1), 4)
        repeated = diagnostics.stratified_split(y[sample])
        np.testing.assert_array_equal(train, repeated[0])
        np.testing.assert_array_equal(validation, repeated[1])

    def test_training_only_medians_scaling_and_all_missing_fallback(self):
        """Imputation and scaling use training rows only; all-missing columns get a fallback."""
        train = np.array([[0.0, np.nan, 7], [2, np.nan, 7], [np.nan, np.nan, 7]])
        validation = np.array([[1000.0, 999, 9], [np.nan, np.nan, np.nan]])
        fitted, records = diagnostics.fit_preprocessing(train, ['value', 'empty', 'constant'])
        np.testing.assert_allclose(fitted['medians'], [1, 0, 7])
        np.testing.assert_allclose(fitted['means'], [1, 0, 7])
        transformed = diagnostics.transform_predictors(validation, fitted)
        self.assertAlmostEqual(transformed[0, 1], 999 / np.std([0, 2, 1]))
        self.assertEqual(transformed[1, 1], 0)
        np.testing.assert_array_equal(transformed[:, 2:], 0)
        np.testing.assert_array_equal(transformed[:, 0], 1)
        self.assertTrue(records[1]['all_missing_training_flag'])
        self.assertTrue(records[2]['zero_variance_training_flag'])
        self.assertTrue(np.isnan(train[2, 0]))
        self.assertEqual(validation[0, 0], 1000)

    def test_metrics_confusion_counts_and_zero_denominators(self):
        """Metrics match known confusion counts and return 0 for empty denominators."""
        y = np.array([1, 1, -1, -1, -1])
        metrics = diagnostics.classification_metrics(y, np.array([1, -1, 1, -1, -1]))
        self.assertEqual(metrics['true_positive'], 1)
        self.assertEqual(metrics['false_negative'], 1)
        self.assertEqual(metrics['false_positive'], 1)
        self.assertEqual(metrics['true_negative'], 2)
        self.assertEqual(metrics['accuracy'], 0.6)
        self.assertAlmostEqual(metrics['balanced_accuracy'], (0.5 + 2 / 3) / 2)
        self.assertEqual(metrics['precision'], 0.5)
        self.assertEqual(metrics['f1'], 0.5)
        reference = diagnostics.classification_metrics(y, -np.ones_like(y))
        self.assertEqual(reference['balanced_accuracy'], 0.5)
        self.assertEqual(reference['precision'], 0)
        self.assertEqual(reference['recall'], 0)

    def test_numeric_response_codes_remain_observed(self):
        """Survey codes such as 7/9 stay observed values, not missing, in the baseline."""
        x = np.array([[7.0], [9.0], [99.0], [np.nan]])
        fitted, records = diagnostics.fit_preprocessing(x, ['survey_code'])
        self.assertEqual(fitted['medians'][0], 9)
        self.assertEqual(records[0]['training_missing_count'], 1)
        self.assertEqual(fitted['means'][0], np.mean([7, 9, 99, 9]))

    def test_loader_checks_all_ids_and_preserves_sample_order(self):
        """Loader validates every ID and returns sampled rows in sample order."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'x_train.csv'
            path.write_text('value,Id,other\n7,a,\n9,b,3\n99,c,4\n', encoding='utf-8')
            names, sample, integrity = diagnostics.load_sampled_predictors(path, ['a', 'b', 'c'], np.array([2, 0]))
            self.assertEqual(names, ['value', 'other'])
            np.testing.assert_allclose(sample, [[99, 4], [7, np.nan]], equal_nan=True)
            self.assertTrue(integrity['exact_id_order'])
            with self.assertRaises(ValueError):
                diagnostics.load_sampled_predictors(path, ['b', 'a', 'c'], np.array([2]))

    def test_singular_mandatory_solve_is_recorded_without_replacement(self):
        """A singular least-squares solve is recorded as failed, not replaced."""
        tx = np.column_stack([np.ones(8), np.zeros(8)])
        y = np.tile([-1.0, 1], 4)
        config = {'config_id': 'singular', 'model': 'least_squares', 'lambda': None,
                  'gamma': None, 'max_iters': None}
        row, predictions = diagnostics.run_configuration(config, tx, tx, y, y)
        self.assertEqual(row['status'], 'failed')
        self.assertIn('LinAlgError', row['error_message'])
        self.assertTrue(np.isnan(row['balanced_accuracy']))
        self.assertEqual(predictions['status'], 'failed')

    def test_all_six_imported_functions_run_with_correct_targets_and_losses(self):
        """All six methods run with the right label coding and report their losses."""
        rng = np.random.default_rng(433)
        tx = np.column_stack([np.ones(100), rng.normal(size=(100, 3))])
        y = np.where(tx[:, 1] > 0, 1.0, -1.0)
        for name in diagnostics.METHODS:
            config = {'config_id': name, 'model': name, 'lambda': 0.01 if 'ridge' in name or name.startswith('reg_') else None,
                      'gamma': 0.01 if name in diagnostics.ITERATIVE_METHODS else None,
                      'max_iters': 5 if name in diagnostics.ITERATIVE_METHODS else None}
            with self.subTest(model=name):
                row, predictions = diagnostics.run_configuration(config, tx[:80], tx[80:], y[:80], y[80:])
                self.assertEqual(row['status'], 'ok', row['error_message'])
                self.assertTrue(row['loss_matches_returned'])
                self.assertTrue(row['finite_weights'])
                self.assertEqual(row['target_representation'], '{0,1}' if name in diagnostics.LOGISTIC_METHODS else '{-1,+1}')
                self.assertEqual(predictions['finite_score_count'], 20)

    def test_restarted_sgd_is_deterministic_and_restores_rng(self):
        """Seeded SGD runs repeat exactly and leave the global RNG state unchanged."""
        tx = np.column_stack([np.ones(20), np.arange(20) / 20])
        y = np.tile([-1.0, 1], 10)
        config = {'config_id': 'sgd', 'model': 'mean_squared_error_sgd', 'lambda': None,
                  'gamma': 0.1, 'max_iters': 40}
        np.random.seed(123)
        state = np.random.get_state()
        first, _ = diagnostics.run_configuration(config, tx, tx, y, y)
        actual_next_draw = np.random.rand()
        np.random.set_state(state)
        self.assertEqual(actual_next_draw, np.random.rand())
        second, _ = diagnostics.run_configuration(config, tx, tx, y, y)
        for key in ('train_loss', 'weight_norm', 'balanced_accuracy', 'f1'):
            self.assertEqual(first[key], second[key])

    def test_convergence_changes_compare_matching_model_and_penalty(self):
        """Convergence comparisons pair runs with the same model and penalty."""
        first = {'model': 'logistic_regression', 'lambda': None, 'max_iters': 10,
                 'status': 'ok', 'train_loss': 0.5, 'training_objective': 0.5,
                 'balanced_accuracy': 0.6}
        second = dict(first, max_iters=50, train_loss=0.4, training_objective=0.4, balanced_accuracy=0.55)
        records = diagnostics.summarize_convergence([second, first])
        self.assertIsNone(records[0]['relative_objective_change'])
        self.assertEqual(records[1]['previous_successful_max_iters'], 10)
        self.assertAlmostEqual(records[1]['data_loss_change'], -0.1)
        self.assertAlmostEqual(records[1]['relative_objective_change'], -0.2)
        self.assertAlmostEqual(records[1]['balanced_accuracy_change'], -0.05)
        self.assertTrue(records[1]['materially_changing_flag'])

    def test_end_to_end_creates_only_separate_outputs(self):
        """A full run writes only to its own output folder."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data, output = root / 'data', root / 'results'
            data.mkdir()
            rng = np.random.default_rng(433)
            ids = np.arange(200)
            x = rng.normal(size=(200, 3))
            x[::10, 1] = np.nan
            y = np.where(x[:, 0] > 0.7, 1, -1)
            np.savetxt(data / 'x_train.csv', np.column_stack([ids, x]), delimiter=',', header='Id,a,b,c', comments='')
            np.savetxt(data / 'y_train.csv', np.column_stack([ids, y]), delimiter=',', header='Id,_MICHD', comments='')
            before = {path.name: path.read_bytes() for path in data.iterdir()}
            with patch.object(diagnostics, 'BATCH_ITERATIONS', (2, 4)), patch.object(diagnostics, 'SGD_ITERATIONS', (10, 20)):
                with contextlib.redirect_stdout(io.StringIO()):
                    diagnostics.main(['--data-dir', str(data), '--output-dir', str(output), '--max-rows', '100'])
            with (output / 'model_metrics.csv').open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row['model'] for row in rows}, set(diagnostics.METHODS) | {'always_negative'})
            self.assertEqual(len(list((output / 'figures').glob('*.png'))), 5)
            report = (output / 'baseline_report.md').read_text(encoding='utf-8')
            self.assertIn('## 14. Implications for model selection', report)
            self.assertTrue(all(path.read_bytes() == before[path.name] for path in data.iterdir()))


if __name__ == '__main__':
    unittest.main()
