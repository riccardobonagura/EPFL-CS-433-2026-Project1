"""Small real-data-path fixtures verify orchestration, isolation and local exports."""

import contextlib
import copy
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from feature_processing.io_utils import load_training_partitions
from ml_attempts.common import best, metric_values, selection_metric
from ml_attempts.confirmation import final_refit, write_submission
from ml_attempts.data import load_context, stratified_folds
from ml_attempts.orchestration import ATTEMPTS, execute, load_configs
from ml_attempts.plots import chart
from ml_attempts.reporting import refine_mse_report
from ml_attempts.representations import base_recipes, fit_representation, load_representation, save_representation, transform_representation


def fixture(root, n=2400):
    """Write a small synthetic dataset folder for end-to-end tests."""
    folder = root / 'dataset'
    folder.mkdir()
    rng = np.random.default_rng(433)
    age = 18 + np.arange(n) % 63
    education = 1 + np.arange(n) % 8
    income = 1 + (np.arange(n) // 8) % 8
    health = 1 + (np.arange(n) // 5) % 5
    bmi = 18 + rng.random(n) * 20
    probability = 1 / (1 + np.exp(-(-2.5 + age / 40 + 0.2 * (education == 6))))
    y = np.where(rng.random(n) < probability, 1, -1)
    x = np.column_stack([age, health, education, income, bmi, np.arange(n), np.arange(n), np.full(n, np.nan)])
    names = ['_AGE80', 'GENHLTH', 'EDUCA', 'INCOME2', '_BMI5', 'SEQNO', '_PSU', 'HAREHAB1']
    with (folder / 'x_train.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Id'] + names)
        writer.writerows([index] + ['' if np.isnan(value) else value for value in row] for index, row in enumerate(x))
    with (folder / 'y_train.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Id', '_MICHD'])
        writer.writerows(enumerate(y))
    with (folder / 'x_test.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Id'] + names)
        writer.writerows([[name, 30 + i * 20, 2, 3, 4, 25, 1, 1, ''] for i, name in enumerate(['second', 'first'])])
    return x, y, names


def tiny_configs():
    """Shrink every attempt's search so the full pipeline runs in seconds."""
    configs = copy.deepcopy(load_configs())
    configs[ATTEMPTS[0]].update(lambda_values=[0.001, 0.01], gamma_values=[0.03, 0.1],
                               batch_size=32, max_updates=6, eval_every=2, patience=2)
    configs[ATTEMPTS[1]].update(competitive_margin=1.0, competitive_alpha_values=[5.0],
                               candidate_priority=['EDUCA', 'INCOME2'], max_recovered_features=2)
    configs[ATTEMPTS[2]].update(gamma_values=[0.001, 0.01], initial_batch_size=32,
                               batch_sizes=[1, 32, 64], max_updates=6,
                               checkpoint_budgets=[2, 8], eval_every=2, patience=2)
    configs[ATTEMPTS[3]].update(lambda_values=[0.001, 0.01], max_iters=20)
    configs[ATTEMPTS[4]].update(lambda_values=[0.001, 0.01], max_iters=20,
                               interactions={'age_x_fair_health': ['_AGE80__scaled', 'GENHLTH__4']})
    return configs


class ExperimentTests(unittest.TestCase):
    """Checks of the experiment pipeline, including a tiny end-to-end run."""
    def test_shared_indices_match_existing_preprocessing_split(self):
        """The shared split equals the preprocessing split for the same seeds."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root, 600)
            context = load_context(root)
            previous = load_training_partitions(root / 'dataset')
            np.testing.assert_array_equal(context['ids'][context['train']], previous['ids_train'])
            np.testing.assert_array_equal(context['ids'][context['validation']], previous['ids_validation'])
            again = load_context(root)
            np.testing.assert_array_equal(context['train'], again['train'])
            self.assertFalse(np.intersect1d(context['train'], context['validation']).size)

    def test_outer_folds_are_stratified_disjoint_and_cover_every_row(self):
        """The 3 folds are disjoint, stratified and cover every row."""
        y = np.r_[-np.ones(90), np.ones(30)]
        folds = stratified_folds(y)
        np.testing.assert_array_equal(np.sort(np.concatenate(folds)), np.arange(len(y)))
        self.assertTrue(all(np.mean(y[fold] == 1) == 0.25 for fold in folds))

    def test_selective_recovery_is_oof_and_quarantine_remains_excluded(self):
        """TE2 recovery is out-of-fold and never uses quarantined fields."""
        with tempfile.TemporaryDirectory() as temporary:
            x, y, names = fixture(Path(temporary))
            recipe = copy.deepcopy(base_recipes()['R0 minimal'])
            recipe.update(kind='recovery', name='TE2', alpha=20.0, recovery=tiny_configs()[ATTEMPTS[1]])
            first, state, _ = fit_representation(x, y, names, recipe)
            self.assertEqual(state['recovered_features'], ['EDUCA', 'INCOME2'])
            changed = y.copy()
            changed[0] = -changed[0]
            second, _, _ = fit_representation(x, changed, names, recipe)
            np.testing.assert_array_equal(first[0], second[0])
            self.assertNotIn('HAREHAB1', state['recovered_features'])
            self.assertEqual(transform_representation(x[:2], names, state).shape[1], first.shape[1])

    def test_selection_metric_orders_runs(self):
        """Runs are ranked by the configured selection metric."""
        rows = [dict(run_id='ba', status='ok', balanced_accuracy=0.80, f1=0.36, recall=0.8, precision=0.2, actual_updates=5),
                dict(run_id='f1', status='ok', balanced_accuracy=0.74, f1=0.42, recall=0.5, precision=0.3, actual_updates=5)]
        self.assertEqual(best(rows)['run_id'], 'ba')
        self.assertEqual(best(rows, 'f1')['run_id'], 'f1')
        with self.assertRaises(ValueError):
            selection_metric({'selection_metric': 'accuracy'})

    def test_metrics_share_augmented_definitions_and_pm1_thresholds(self):
        """Pipeline metrics match the augmented definitions for {-1, +1} labels."""
        metrics = metric_values(np.array([-1, -1, 1, 1]), np.array([-2.0, 0.5, 0.2, 1]), 0)
        self.assertEqual(metrics['tp'], 2)
        self.assertEqual(metrics['fp'], 1)
        self.assertEqual(metrics['balanced_accuracy'], 0.75)
        self.assertEqual(metrics['predicted_positive_fraction'], 0.75)

    def test_saved_recovery_representation_reloads_without_training_data(self):
        """A saved representation transforms new rows without training data."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            x, y, names = fixture(root)
            recipe = copy.deepcopy(base_recipes()['R0 minimal'])
            recipe.update(kind='recovery', name='TE2', alpha=20.0, recovery=tiny_configs()[ATTEMPTS[1]])
            _, state, _ = fit_representation(x, y, names, recipe)
            save_representation(state, root / 'saved_state')
            restored = load_representation(root / 'saved_state')
            np.testing.assert_array_equal(transform_representation(x[:10], names, state),
                                          transform_representation(x[:10], names, restored))

    def test_final_export_requires_confirmation_before_test_access(self):
        """The test file is read only after confirmation is complete."""
        with patch('ml_attempts.confirmation.load_unlabeled_csv') as loader:
            with self.assertRaises(ValueError):
                final_refit({}, {'confirmation_complete': False})
            loader.assert_not_called()

    def test_submission_order_and_label_validation(self):
        """Submissions keep ID order and only contain -1/+1 labels."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'submission.csv'
            write_submission(path, ['z', 'a'], np.array([1, -1]))
            with path.open(newline='', encoding='utf-8') as stream:
                self.assertEqual(list(csv.reader(stream)), [['Id', 'Prediction'], ['z', '1'], ['a', '-1']])
            with self.assertRaises(ValueError):
                write_submission(path, ['z', 'a'], np.array([0, 1]))

    def test_stdlib_numpy_figures_have_valid_png_and_svg_headers(self):
        """NumPy-drawn charts are valid PNG and SVG files."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'example.png'
            chart(path, 'Example', ['A', 'B'], {'BA': [0.5, 0.8]}, 'Stage', 'Balanced accuracy', unit_interval=True)
            self.assertTrue(path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
            self.assertIn('<svg', path.with_suffix('.svg').read_text(encoding='utf-8'))

    def test_mse_interpretation_detects_selected_threshold_decline(self):
        """The MSE report flags when a longer budget lowers the selected metric."""
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            rows = [dict(status='ok', stage='classification_checkpoint', max_updates=4000,
                         train_loss=0.2, default_balanced_accuracy=0.51, balanced_accuracy=0.78),
                    dict(status='ok', stage='classification_checkpoint', max_updates=8000,
                         train_loss=0.15, default_balanced_accuracy=0.52, balanced_accuracy=0.77)]
            for filename in ['all_runs.csv', 'convergence_results.csv']:
                with (output / filename).open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            (output / 'experiment_report.md').write_text('# Example\n\n## Selection limits and completion\n', encoding='utf-8')
            refine_mse_report(output)
            report = (output / 'experiment_report.md').read_text(encoding='utf-8')
            self.assertIn('4000 -> 8000', report)
            self.assertIn('78.000% -> 77.000%', report)

    def test_all_attempts_confirmation_and_full_refits_end_to_end(self):
        """All five attempts run search, confirmation and refit end to end."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            sources = {path.name: path.read_bytes() for path in (root / 'dataset').iterdir()}
            context = load_context(root)
            from ml_attempts.confirmation import load_unlabeled_csv
            def guarded_test_loader(path):
                """Fail if the test file is read before every attempt is confirmed."""
                for name in ATTEMPTS:
                    selection = json.loads((root / 'ml_attempts' / name / 'outputs/selection.json').read_text(encoding='utf-8'))
                    self.assertTrue(selection['confirmation_complete'])
                return load_unlabeled_csv(path)
            with contextlib.redirect_stdout(io.StringIO()), patch('ml_attempts.confirmation.load_unlabeled_csv', side_effect=guarded_test_loader):
                results = execute(context, tiny_configs(), ATTEMPTS, phase='all')
            self.assertEqual(set(results), set(ATTEMPTS))
            protocols = set()
            for attempt in ATTEMPTS:
                output = root / 'ml_attempts' / attempt / 'outputs'
                payload = json.loads((output / 'selection.json').read_text(encoding='utf-8'))
                self.assertTrue(payload['confirmation_complete'])
                self.assertEqual(payload['final_refit']['training_rows'], 2400)
                protocols.add(payload['confirmation']['protocol_sha256'])
                expected_metric = 'f1'
                self.assertEqual(payload['selection_metric'], expected_metric)
                self.assertEqual(payload['confirmation']['threshold_metric'], expected_metric)
                self.assertEqual(payload['final_refit']['threshold'], payload['confirmation']['threshold'])
                frozen = json.loads((output / 'frozen_final_configuration.json').read_text(encoding='utf-8'))
                self.assertFalse(frozen['params']['early_stopping'])
                for filename in ['validation_metrics.csv', 'all_runs.csv', 'threshold_results.csv',
                                 'experiment_report.md', 'run_config.txt', 'final_model_weights.npz',
                                 'test_predictions.csv', 'submission.csv', 'cv_metrics.csv']:
                    self.assertTrue((output / filename).is_file(), (attempt, filename))
                state = json.loads((output / 'preprocessing_state/preprocessing_state.json').read_text(encoding='utf-8'))
                self.assertEqual(state['training_rows'], 2400)
                for fold in range(1, 4):
                    with np.load(output / f'confirmation/fold_{fold}/indices.npz') as indices:
                        self.assertFalse(np.intersect1d(indices['training'], indices['validation']).size)
                        self.assertFalse(np.intersect1d(indices['stopping'], indices['validation']).size)
                        fold_state = json.loads((output / f'confirmation/fold_{fold}/preprocessing_state/preprocessing_state.json').read_text(encoding='utf-8'))
                        self.assertEqual(fold_state['training_rows'], len(indices['training']))
                with (output / 'submission.csv').open(newline='', encoding='utf-8') as stream:
                    submission = list(csv.reader(stream))
                self.assertEqual([row[0] for row in submission[1:]], ['second', 'first'])
                self.assertTrue(all(row[1] in {'-1', '1'} for row in submission[1:]))
            self.assertEqual(len(protocols), 1)
            self.assertTrue((root / 'ml_attempts/model_comparison.csv').is_file())
            self.assertTrue(all(path.read_bytes() == sources[path.name] for path in (root / 'dataset').iterdir()))


if __name__ == '__main__':
    unittest.main()
