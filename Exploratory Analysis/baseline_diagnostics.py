"""Separate diagnostics of the six mandatory Project 1 functions.

This module does not import or execute eda.py, read EDA outputs, modify course
implementations, choose final features, or optimize classification thresholds.
"""

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
IMPLEMENTATIONS_PATH = PROJECT_DIR / 'implementations.py'
sys.path.insert(0, str(PROJECT_DIR))
from implementations import (
    mean_squared_error_gd,
    mean_squared_error_sgd,
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
)


RANDOM_SEED = 42
VALIDATION_FRACTION = 0.20
MAX_DIAGNOSTIC_ROWS = 100_000
BATCH_ITERATIONS = (10, 50, 100)
SGD_ITERATIONS = (1_000, 10_000, 100_000)
RIDGE_LAMBDAS = (0.0, 1e-4, 1e-2, 1.0)
LOGISTIC_LAMBDAS = (1e-4, 1e-2, 1.0)
ZERO_STD_TOLERANCE = 1e-12
COLLAPSE_POSITIVE_FRACTION = 0.01
METHODS = {
    'mean_squared_error_gd': mean_squared_error_gd,
    'mean_squared_error_sgd': mean_squared_error_sgd,
    'least_squares': least_squares,
    'ridge_regression': ridge_regression,
    'logistic_regression': logistic_regression,
    'reg_logistic_regression': reg_logistic_regression,
}
ITERATIVE_METHODS = {
    'mean_squared_error_gd', 'mean_squared_error_sgd',
    'logistic_regression', 'reg_logistic_regression',
}
LOGISTIC_METHODS = {'logistic_regression', 'reg_logistic_regression'}


def sha256_file(path):
    """Return the SHA-256 hex digest of a file, read in 1 MB blocks."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_targets(path):
    """Read target IDs as exact text, independently of the predictor CSV."""
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if len(header) != 2 or set(header) != {'Id', '_MICHD'}:
            raise ValueError('y_train.csv must contain exactly Id and _MICHD')
        id_index, target_index = header.index('Id'), header.index('_MICHD')
        ids, labels = [], []
        for row in reader:
            if len(row) != len(header):
                raise ValueError('Malformed target CSV row')
            ids.append(row[id_index].strip())
            labels.append(float(row[target_index]))
    if not ids or any(not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError('Target IDs must be present and unique')
    y_pm1 = np.asarray(labels, dtype=float)
    if not np.array_equal(np.unique(y_pm1), [-1, 1]):
        raise ValueError('Expected both target classes, exactly {-1,+1}')
    return ids, y_pm1


def stratified_subsample(y_pm1, max_rows, seed=RANDOM_SEED):
    """Allocate exact sample size with proportional largest-remainder rounding."""
    if max_rows < 4:
        raise ValueError('The diagnostic sample needs at least four rows')
    classes, counts = np.unique(y_pm1, return_counts=True)
    if len(classes) != 2 or min(counts) < 2:
        raise ValueError('Each target class needs at least two rows')
    size = min(len(y_pm1), max_rows)
    quotas = counts * (size / len(y_pm1))
    allocated = np.floor(quotas).astype(int)
    for index in np.argsort(-(quotas - allocated), kind='stable')[:size - allocated.sum()]:
        allocated[index] += 1
    if min(allocated) < 2:
        raise ValueError('Increase sample size to retain at least two rows per class')
    rng = np.random.default_rng(seed)
    selected = [rng.permutation(np.flatnonzero(y_pm1 == label))[:count]
                for label, count in zip(classes, allocated)]
    return rng.permutation(np.concatenate(selected))


def stratified_split(y_pm1, validation_fraction=VALIDATION_FRACTION, seed=RANDOM_SEED + 1):
    """Split row indices into class-balanced training/validation parts with a fixed seed."""
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be strictly between 0 and 1')
    rng = np.random.default_rng(seed)
    training, validation = [], []
    for label in (-1, 1):
        indices = rng.permutation(np.flatnonzero(y_pm1 == label))
        if len(indices) < 2:
            raise ValueError('Each class needs training and validation rows')
        count = max(1, min(len(indices) - 1, int(round(len(indices) * validation_fraction))))
        validation.append(indices[:count])
        training.append(indices[count:])
    return rng.permutation(np.concatenate(training)), rng.permutation(np.concatenate(validation))


def load_sampled_predictors(path, target_ids, sampled_indices):
    """Validate every input ID, but parse predictor numbers only for sampled rows."""
    selected_position = {int(source): position for position, source in enumerate(sampled_indices)}
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if 'Id' not in header or len(set(header)) != len(header) or len(header) < 2:
            raise ValueError('X requires Id, unique column names, and at least one predictor')
        id_index = header.index('Id')
        columns = [index for index, name in enumerate(header) if name != 'Id']
        names = [header[index] for index in columns]
        sample = np.empty((len(sampled_indices), len(columns)), dtype=float)
        seen, missing_ids, duplicate_ids, mismatch_count, rows = set(), 0, 0, 0, 0
        missing_tokens = {'', 'nan', 'na'}
        for source, row in enumerate(reader):
            if len(row) != len(header):
                raise ValueError(f'x_train.csv row {source + 2}: wrong number of fields')
            rows += 1
            record_id = row[id_index].strip()
            missing_ids += int(not record_id)
            duplicate_ids += int(record_id in seen)
            seen.add(record_id)
            if source >= len(target_ids) or record_id != target_ids[source]:
                mismatch_count += 1
            if source in selected_position:
                values = np.fromiter(
                    (np.nan if row[index].strip().lower() in missing_tokens else float(row[index])
                     for index in columns), dtype=float, count=len(columns),
                )
                if np.isinf(values).any():
                    raise ValueError('Infinite sampled predictor values are not supported')
                sample[selected_position[source]] = values
    integrity = {
        'x_rows': rows, 'y_rows': len(target_ids), 'predictors_excluding_Id': len(names),
        'x_missing_ids': missing_ids, 'x_duplicate_ids': duplicate_ids,
        'y_missing_ids': 0, 'y_duplicate_ids': 0,
        'same_id_set': seen == set(target_ids),
        'exact_id_order': mismatch_count == 0 and rows == len(target_ids),
    }
    if missing_ids or duplicate_ids or not integrity['same_id_set'] or not integrity['exact_id_order']:
        raise ValueError('Training ID integrity/alignment failure: ' + json.dumps(integrity))
    return names, sample, integrity


def fit_preprocessing(x_train, names):
    """Fit median, imputed mean, and imputed std on training rows exclusively."""
    medians = np.empty(x_train.shape[1])
    all_missing = np.zeros(x_train.shape[1], dtype=bool)
    for index in range(x_train.shape[1]):
        observed = x_train[np.isfinite(x_train[:, index]), index]
        all_missing[index] = len(observed) == 0
        medians[index] = np.median(observed) if len(observed) else 0.0
    filled = np.where(np.isnan(x_train), medians, x_train)
    means, stds = filled.mean(axis=0), filled.std(axis=0)
    zero_variance = stds <= ZERO_STD_TOLERANCE
    scales = np.where(zero_variance, 1.0, stds)
    records = [{
        'feature_name': name, 'training_missing_count': int(np.isnan(x_train[:, i]).sum()),
        'training_missing_fraction': float(np.isnan(x_train[:, i]).mean()),
        'training_median': float(medians[i]), 'training_imputed_mean': float(means[i]),
        'training_imputed_std': float(stds[i]), 'applied_scale': float(scales[i]),
        'all_missing_training_flag': bool(all_missing[i]),
        'zero_variance_training_flag': bool(zero_variance[i]),
    } for i, name in enumerate(names)]
    return {'medians': medians, 'means': means, 'scales': scales,
            'zero_variance': zero_variance, 'all_missing': all_missing}, records


def transform_predictors(x, fitted):
    """Apply training statistics and prepend an explicit intercept column."""
    tx = np.empty((len(x), x.shape[1] + 1), dtype=float)
    tx[:, 0] = 1.0
    transformed = tx[:, 1:]
    np.copyto(transformed, x)
    missing = np.isnan(transformed)
    np.copyto(transformed, np.broadcast_to(fitted['medians'], transformed.shape), where=missing)
    transformed -= fitted['means']
    transformed /= fitted['scales']
    # A constant training feature has no estimable numeric variation. Preserve
    # its column as zero in both splits; do not amplify validation deviations.
    transformed[:, fitted['zero_variance']] = 0.0
    if not np.isfinite(tx).all():
        raise ValueError('Nonfinite transformed predictor encountered')
    return tx


def classification_metrics(y_pm1, predicted_pm1):
    """Confusion counts and metrics; undefined precision/recall ratios become 0."""
    if y_pm1.shape != predicted_pm1.shape:
        raise ValueError('Prediction shape must match target shape')
    if not np.isin(predicted_pm1, [-1, 1]).all():
        raise ValueError('Predicted classes must be {-1,+1}')
    positive, predicted = y_pm1 == 1, predicted_pm1 == 1
    tp = int(np.sum(positive & predicted))
    tn = int(np.sum(~positive & ~predicted))
    fp = int(np.sum(~positive & predicted))
    fn = int(np.sum(positive & ~predicted))
    divide = lambda numerator, denominator: numerator / denominator if denominator else 0.0
    recall, specificity = divide(tp, tp + fn), divide(tn, tn + fp)
    return {
        'accuracy': divide(tp + tn, len(y_pm1)), 'balanced_accuracy': (recall + specificity) / 2,
        'precision': divide(tp, tp + fp), 'recall': recall, 'specificity': specificity,
        'f1': divide(2 * tp, 2 * tp + fp + fn),
        'true_positive': tp, 'true_negative': tn, 'false_positive': fp, 'false_negative': fn,
        'actual_positive_fraction': float(positive.mean()),
        'predicted_positive_fraction': float(predicted.mean()),
    }


def data_loss(y, tx, w, logistic):
    """Unregularized loss: mean logistic NLL if logistic, else half-MSE."""
    scores = tx @ w
    if logistic:
        return float(np.mean(np.logaddexp(0.0, np.where(y == 1, -scores, scores))))
    return float(0.5 * np.mean((y - scores)**2))


def choose_configurations(tx):
    """Use training-only curvature bounds, not validation performance, for steps."""
    gram = tx.T @ tx / len(tx)
    eigenvalues = np.linalg.eigvalsh(gram)
    largest = max(float(eigenvalues[-1]), 1e-12)
    rank_tolerance = largest * tx.shape[1] * np.finfo(float).eps
    max_row_norm_sq = max(float(np.einsum('ij,ij->i', tx, tx).max()), 1.0)
    gd_gamma, sgd_gamma = 1.0 / largest, 1.0 / max_row_norm_sq
    configs = []
    for iterations in BATCH_ITERATIONS:
        configs.append({'model': 'mean_squared_error_gd', 'lambda': None,
                        'gamma': gd_gamma, 'max_iters': iterations})
    for iterations in SGD_ITERATIONS:
        configs.append({'model': 'mean_squared_error_sgd', 'lambda': None,
                        'gamma': sgd_gamma, 'max_iters': iterations})
    configs.append({'model': 'least_squares', 'lambda': None, 'gamma': None, 'max_iters': None})
    for penalty in RIDGE_LAMBDAS:
        configs.append({'model': 'ridge_regression', 'lambda': penalty, 'gamma': None, 'max_iters': None})
    for iterations in BATCH_ITERATIONS:
        configs.append({'model': 'logistic_regression', 'lambda': None,
                        'gamma': 1.0 / (0.25 * largest), 'max_iters': iterations})
    for penalty in LOGISTIC_LAMBDAS:
        for iterations in BATCH_ITERATIONS:
            configs.append({'model': 'reg_logistic_regression', 'lambda': penalty,
                            'gamma': 1.0 / (0.25 * largest + 2 * penalty), 'max_iters': iterations})
    conditioning = {
        'largest_training_gram_eigenvalue': largest,
        'training_gram_dimension': tx.shape[1],
        'training_gram_numerical_rank': int((eigenvalues > rank_tolerance).sum()),
        'training_gram_rank_tolerance': rank_tolerance,
        'max_training_row_squared_norm': max_row_norm_sq,
        'gd_gamma': gd_gamma, 'sgd_gamma': sgd_gamma,
    }
    for index, config in enumerate(configs, 1):
        config['config_id'] = f'run_{index:02d}'
    return configs, conditioning


def run_configuration(config, tx_train, tx_validation, y_train_pm1, y_validation_pm1):
    """Call an unmodified mandatory function and audit its returned result."""
    model = config['model']
    logistic = model in LOGISTIC_METHODS
    y_train_01 = (y_train_pm1 + 1) / 2
    target = y_train_01 if logistic else y_train_pm1
    initial_w = np.zeros(tx_train.shape[1])
    row = {
        'config_id': config['config_id'], 'model': model, 'lambda': config['lambda'],
        'gamma': config['gamma'], 'max_iters': config['max_iters'],
        'n_train': len(tx_train), 'n_validation': len(tx_validation), 'n_parameters': tx_train.shape[1],
        'target_representation': '{0,1}' if logistic else '{-1,+1}',
        'decision_threshold': 0.5 if logistic else 0.0,
        'threshold_space': 'probability' if logistic else 'linear_score',
        'status': 'failed', 'error_message': '', 'train_loss': np.nan,
        'audited_train_loss': np.nan, 'loss_matches_returned': False,
        'training_objective': np.nan, 'initial_train_loss': float(np.log(2)) if logistic else 0.5,
        'weight_norm': np.nan, 'finite_weights': False, 'finite_loss': False, 'finite_scores': False,
        'training_runtime_seconds': np.nan, 'prediction_runtime_seconds': np.nan,
        'accuracy': np.nan, 'balanced_accuracy': np.nan, 'precision': np.nan,
        'recall': np.nan, 'specificity': np.nan, 'f1': np.nan,
        'true_positive': None, 'true_negative': None, 'false_positive': None, 'false_negative': None,
        'actual_positive_fraction': float(np.mean(y_validation_pm1 == 1)),
        'predicted_positive_fraction': np.nan, 'near_all_negative_flag': False,
        'near_all_positive_flag': False, 'constant_score_flag': False,
        'training_data_passes': (config['max_iters'] / len(tx_train) if model == 'mean_squared_error_sgd'
                                 else config['max_iters']) if model in ITERATIVE_METHODS else None,
    }
    prediction = {
        'config_id': config['config_id'], 'model': model, 'status': 'failed',
        'score_minimum': np.nan, 'score_maximum': np.nan, 'score_mean': np.nan,
        'score_std': np.nan, 'probability_minimum': np.nan,
        'probability_maximum': np.nan, 'probability_mean': np.nan,
        'predicted_positive_fraction': np.nan, 'finite_score_count': 0,
        'validation_rows': len(tx_validation),
    }
    rng_state = np.random.get_state()
    np.random.seed(RANDOM_SEED + 1001)  # Restart SGD so iteration budgets share a sampling prefix.
    training_start = time.perf_counter()
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            method = METHODS[model]
            if model == 'least_squares':
                w, loss = method(target, tx_train)
            elif model == 'ridge_regression':
                w, loss = method(target, tx_train, config['lambda'])
            elif model == 'reg_logistic_regression':
                w, loss = method(target, tx_train, config['lambda'], initial_w,
                                 config['max_iters'], config['gamma'])
            else:
                w, loss = method(target, tx_train, initial_w, config['max_iters'], config['gamma'])
        row['training_runtime_seconds'] = time.perf_counter() - training_start
        if w.shape != (tx_train.shape[1],) or not np.isscalar(loss):
            raise ValueError('Mandatory return contract violated')
        row.update(train_loss=float(loss), finite_weights=bool(np.isfinite(w).all()),
                   finite_loss=bool(np.isfinite(loss)))
        if not row['finite_weights'] or not row['finite_loss']:
            raise FloatingPointError('Nonfinite weights or returned loss')
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            row['weight_norm'] = float(np.linalg.norm(w))
            row['audited_train_loss'] = data_loss(target, tx_train, w, logistic)
            row['loss_matches_returned'] = bool(np.isclose(loss, row['audited_train_loss'], rtol=1e-10, atol=1e-12))
            penalty = config['lambda'] or 0.0
            row['training_objective'] = float(loss + penalty * (w @ w))
            prediction_start = time.perf_counter()
            scores = tx_validation @ w
            probability = np.exp(-np.logaddexp(0.0, -scores)) if logistic else None
            predicted = np.where(probability >= 0.5 if logistic else scores >= 0, 1, -1)
            row['prediction_runtime_seconds'] = time.perf_counter() - prediction_start
        row['finite_scores'] = bool(np.isfinite(scores).all())
        if not row['finite_scores']:
            raise FloatingPointError('Nonfinite validation scores')
        if not row['loss_matches_returned']:
            raise ValueError('Returned loss does not match recomputed unregularized training loss')
        row.update(classification_metrics(y_validation_pm1, predicted))
        row['near_all_negative_flag'] = row['predicted_positive_fraction'] < COLLAPSE_POSITIVE_FRACTION
        row['near_all_positive_flag'] = row['predicted_positive_fraction'] > 1 - COLLAPSE_POSITIVE_FRACTION
        row['constant_score_flag'] = bool(scores.std() <= 1e-12)
        row['status'] = 'ok'
        prediction.update({
            'status': 'ok', 'score_minimum': float(scores.min()), 'score_maximum': float(scores.max()),
            'score_mean': float(scores.mean()), 'score_std': float(scores.std()),
            'predicted_positive_fraction': row['predicted_positive_fraction'],
            'finite_score_count': int(np.isfinite(scores).sum()),
        })
        if logistic:
            prediction.update(probability_minimum=float(probability.min()),
                              probability_maximum=float(probability.max()),
                              probability_mean=float(probability.mean()))
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError, ValueError) as error:
        if not np.isfinite(row['training_runtime_seconds']):
            row['training_runtime_seconds'] = time.perf_counter() - training_start
        row['error_message'] = f'{type(error).__name__}: {error}'
    finally:
        np.random.set_state(rng_state)
    return row, prediction


def majority_baseline(y_train_pm1, y_validation_pm1, n_parameters):
    """Metrics row for the always-negative reference classifier."""
    row = {
        'config_id': 'reference', 'model': 'always_negative', 'lambda': None, 'gamma': None,
        'max_iters': None, 'n_train': len(y_train_pm1), 'n_validation': len(y_validation_pm1),
        'n_parameters': n_parameters, 'status': 'reference', 'error_message': '',
        'train_loss': np.nan, 'weight_norm': np.nan,
        'training_runtime_seconds': 0.0, 'prediction_runtime_seconds': 0.0,
        'finite_weights': None, 'finite_loss': None, 'finite_scores': True,
    }
    row.update(classification_metrics(y_validation_pm1, -np.ones_like(y_validation_pm1)))
    return row


def summarize_convergence(rows):
    """Compare independent budgets within each fixed model/lambda configuration."""
    output = []
    groups = {}
    for row in rows:
        if row['model'] in ITERATIVE_METHODS:
            groups.setdefault((row['model'], row['lambda']), []).append(row)
    for group in groups.values():
        preceding = None
        for row in sorted(group, key=lambda record: record['max_iters']):
            record = dict(row)
            record.update(previous_successful_max_iters=None, data_loss_change=None,
                          relative_objective_change=None, balanced_accuracy_change=None,
                          materially_changing_flag=None)
            if row['status'] == 'ok':
                if preceding is not None:
                    objective_change = row['training_objective'] - preceding['training_objective']
                    relative = objective_change / max(abs(preceding['training_objective']), 1e-12)
                    metric_change = row['balanced_accuracy'] - preceding['balanced_accuracy']
                    record.update(
                        previous_successful_max_iters=preceding['max_iters'],
                        data_loss_change=row['train_loss'] - preceding['train_loss'],
                        relative_objective_change=relative,
                        balanced_accuracy_change=metric_change,
                        materially_changing_flag=abs(relative) >= 0.01 or abs(metric_change) >= 0.01,
                    )
                preceding = row
            output.append(record)
    return output


def write_csv(path, records, keys=None):
    """Write a list of dict records as CSV."""
    if keys is None:
        keys = list(dict.fromkeys(key for record in records for key in record))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        for record in records:
            writer.writerow({key: '' if value is None or isinstance(value, (float, np.floating)) and not np.isfinite(value)
                             else value for key, value in record.items()})


def format_value(value):
    """Format a value for a Markdown table cell."""
    if value is None:
        return 'not available'
    if isinstance(value, (float, np.floating)):
        return f'{value:.5g}' if np.isfinite(value) else 'not available'
    return str(value).replace('|', '/').replace('\n', ' ')


def table(rows, columns):
    """Render rows as a Markdown table with the given (key, title) columns."""
    if not rows:
        return 'No successful configurations available.\n'
    lines = ['| ' + ' | '.join(title for _, title in columns) + ' |',
             '| ' + ' | '.join('---' for _ in columns) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join(format_value(row.get(key)) for key, _ in columns) + ' |')
    return '\n'.join(lines) + '\n'


def best_run(rows, models, metric='balanced_accuracy'):
    """Best successful run among the given models, ranked by metric then F1 and recall."""
    candidates = [row for row in rows if row['model'] in models and row['status'] == 'ok']
    return max(candidates, key=lambda r: (r[metric], r['f1'], r['recall'], r['config_id'])) if candidates else None


def describe_run(row):
    """One-sentence summary of a run (model, lambda, BA, F1, recall)."""
    if row is None:
        return 'no successful configuration'
    return (f'`{row["model"]}` ({row["config_id"]}, lambda={row["lambda"]}, iterations={row["max_iters"]}): '
            f'balanced accuracy {row["balanced_accuracy"]:.3%}, F1 {row["f1"]:.3%}, recall {row["recall"]:.3%}')


def regularization_comparison(rows, ordinary_name, regularized_name):
    """Sentence comparing the best unregularized and regularized runs."""
    ordinary = best_run(rows, {ordinary_name})
    regularized = best_run(rows, {regularized_name})
    if ordinary is None:
        return (f'`{ordinary_name}` has no successful run in this representation. The best successful '
                f'regularized result is {describe_run(regularized)}. This supports a numerical feasibility '
                'comparison, not a measured validation improvement over a successful unregularized fit.')
    if regularized is None:
        return f'No successful regularized run; unregularized reference: {describe_run(ordinary)}.'
    delta = regularized['balanced_accuracy'] - ordinary['balanced_accuracy']
    outcome = 'higher' if delta > 1e-4 else 'lower' if delta < -1e-4 else 'essentially unchanged'
    return (f'Best observed regularized balanced accuracy is {outcome}: signed difference {delta * 100:+.3f} '
            f'percentage points. Unregularized: {describe_run(ordinary)}. Regularized: {describe_run(regularized)}. '
            f'F1 difference {regularized["f1"] - ordinary["f1"]:+.5f}; recall difference '
            f'{regularized["recall"] - ordinary["recall"]:+.5f}. These are exploratory selections on one '
            'validation split, with finite iteration budgets and potentially different optimization rates.')


def generate_figures(output, rows):
    """Save matplotlib figures of the baseline diagnostics."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = output / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    selected = [rows[0]] + [row for model in METHODS if (row := best_run(rows, {model})) is not None]
    labels = [row['model'].replace('mean_squared_error_', 'MSE ').replace('_regression', '').replace('_', ' ')
              for row in selected]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

    def save(fig, name):
        """Save one figure as PNG and close it."""
        fig.tight_layout()
        fig.savefig(figures / name, dpi=140, metadata={'Software': 'Mandatory baseline diagnostics'})
        plt.close(fig)

    index = np.arange(len(selected))
    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, metric, label in [(-0.25, 'balanced_accuracy', 'Balanced accuracy'), (0, 'f1', 'F1'), (0.25, 'accuracy', 'Accuracy')]:
        ax.bar(index + offset, [r[metric] for r in selected], width=0.25, label=label)
    ax.set_xticks(index, labels, rotation=25, ha='right')
    ax.set(title='Best observed configuration per successful mandatory method', xlabel='Method (selected by validation balanced accuracy)', ylabel='Metric (fraction)', ylim=(0, 1.05))
    ax.legend()
    save(fig, '01_model_metric_comparison.png')

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(index - 0.2, [r['recall'] for r in selected], width=0.4, label='Recall / sensitivity')
    ax.bar(index + 0.2, [r['specificity'] for r in selected], width=0.4, label='Specificity')
    ax.set_xticks(index, labels, rotation=25, ha='right')
    ax.set(title='Minority recall versus negative-class specificity', xlabel='Successful method / majority reference', ylabel='Metric (fraction)', ylim=(0, 1.05))
    ax.legend()
    save(fig, '02_recall_specificity_comparison.png')

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(index, [r['predicted_positive_fraction'] * 100 for r in selected], color='#3b6c8d')
    ax.axhline(rows[0]['actual_positive_fraction'] * 100, color='#d76e4d', linestyle='--', label='Actual positive prevalence')
    ax.set_xticks(index, labels, rotation=25, ha='right')
    ax.set(title='Default-threshold positive predictions', xlabel='Successful method / majority reference', ylabel='Predicted positive (%)')
    ax.legend()
    save(fig, '03_predicted_positive_rate.png')

    timed = [r for r in rows if r['model'] != 'always_negative']
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh([f'{r["config_id"]} {r["model"]}' for r in timed][::-1],
            [max(r['training_runtime_seconds'], 1e-6) for r in timed][::-1], color='#3b6c8d')
    ax.set_xscale('log')
    ax.set(title='Training runtime for every attempted configuration (failures included)', xlabel='Wall-clock training seconds (log scale)', ylabel='Configuration / method')
    save(fig, '04_runtime_comparison.png')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, model in zip(axes, ['ridge_regression', 'reg_logistic_regression']):
        candidates = [r for r in rows if r['model'] == model and r['status'] == 'ok']
        latest = [max([r for r in candidates if r['lambda'] == penalty], key=lambda r: r['max_iters'] or 0)
                  for penalty in sorted({r['lambda'] for r in candidates})]
        ax.plot([r['lambda'] for r in latest], [r['balanced_accuracy'] for r in latest], 'o-', label='Balanced accuracy')
        ax.plot([r['lambda'] for r in latest], [r['f1'] for r in latest], 's-', label='F1')
        ax.set_xscale('symlog', linthresh=1e-5)
        ax.set(title=model, xlabel='L2 lambda (largest tested iteration budget)', ylabel='Validation metric (fraction)', ylim=(0, 1.05))
        ax.legend()
    save(fig, '05_regularization_comparison.png')


def write_report(output, rows, metadata, preprocessing_records):
    """Write the Markdown report of the six baseline methods."""
    setup = metadata['setup']
    conditioning = metadata['conditioning']
    successful = [row for row in rows if row['status'] == 'ok']
    failed = [row for row in rows if row['status'] == 'failed']
    collapsed = [row for row in successful if row['near_all_negative_flag']]
    linear = best_run(rows, {'mean_squared_error_gd', 'mean_squared_error_sgd', 'least_squares', 'ridge_regression'})
    logistic = best_run(rows, LOGISTIC_METHODS)
    winners = [best_run(rows, {model}) for model in METHODS]
    winners = [row for row in winners if row is not None]
    constants = [r['feature_name'] for r in preprocessing_records if r['zero_variance_training_flag']]
    all_missing = [r['feature_name'] for r in preprocessing_records if r['all_missing_training_flag']]
    convergence = summarize_convergence(rows)
    final_budgets = []
    for model, penalty in dict.fromkeys((r['model'], r['lambda']) for r in convergence):
        group = [r for r in convergence if r['model'] == model and r['lambda'] == penalty and r['status'] == 'ok']
        if group:
            final_budgets.append(max(group, key=lambda r: r['max_iters']))
    optimization_observations = []
    for model in ['mean_squared_error_gd', 'mean_squared_error_sgd', 'logistic_regression']:
        group = sorted([r for r in successful if r['model'] == model], key=lambda r: r['max_iters'])
        if len(group) > 1:
            first, last = group[0], group[-1]
            observation = (
                f'- `{model}` from {first["max_iters"]} to {last["max_iters"]} updates: data loss '
                f'{first["train_loss"]:.6g} -> {last["train_loss"]:.6g}, balanced accuracy '
                f'{first["balanced_accuracy"]:.3%} -> {last["balanced_accuracy"]:.3%}, recall '
                f'{first["recall"]:.3%} -> {last["recall"]:.3%}, runtime '
                f'{first["training_runtime_seconds"]:.3f}s -> {last["training_runtime_seconds"]:.3f}s.'
            )
            if last['train_loss'] < first['train_loss'] and last['balanced_accuracy'] < first['balanced_accuracy']:
                observation += (' Lower training loss coincides with lower balanced accuracy at the fixed threshold; '
                                'optimization progress alone is not minority-class metric progress.')
            optimization_observations.append(observation)
    numeric_columns = [('model', 'Method'), ('config_id', 'Run'), ('max_iters', 'Updates'), ('lambda', 'Lambda'),
                       ('train_loss', 'Data loss'), ('balanced_accuracy', 'Bal. acc.'), ('recall', 'Recall'),
                       ('f1', 'F1'), ('training_runtime_seconds', 'Train seconds')]
    if linear is not None and logistic is not None:
        delta = logistic['balanced_accuracy'] - linear['balanced_accuracy']
        interpretation = ('Logistic has higher' if delta > 1e-4 else 'Regression-based scoring has higher' if delta < -1e-4 else 'Neither family has clearly higher')
        family_answer = (f'{interpretation} observed balanced accuracy in this limited comparison '
                         f'(logistic minus linear: {delta * 100:+.3f} percentage points). '
                         f'Linear: {describe_run(linear)}. Logistic: {describe_run(logistic)}. '
                         'This does not establish a universally preferable objective: thresholds are fixed, '
                         'optimization budgets are finite, and no final categorical preprocessing was used.')
    else:
        family_answer = 'At least one family has no successful configuration, so a numerical family comparison is unavailable.'
    report = [
        '# Mandatory Model Baseline Diagnostics', '',
        '## 1. Experimental setup', '',
        'This is a separate diagnostic, not a final modeling pipeline. It imports all six functions from',
        '`../implementations.py`. It does not import/run/edit `eda.py` or read/write',
        '`eda_outputs/`. The mandatory file is fingerprinted before and after execution and remains unchanged.',
        'No class weighting, one-hot encoding, engineered interactions, feature selection, code recoding,',
        'threshold optimization or extensive hyperparameter search is performed.', '',
        f'Seed **{RANDOM_SEED}**; sampling RNG seed {RANDOM_SEED}; split RNG seed {RANDOM_SEED + 1};',
        f'SGD RNG seed {RANDOM_SEED + 1001}, restarted for each independent configuration.',
        'Every iterative run starts from a fresh one-dimensional zero vector. Different iteration budgets',
        'are separate calls from zero; SGD uses the same sampled-point sequence prefix, not minibatches.',
        'Metrics and predictions are deterministic for identical inputs/configurations; wall-clock timings vary.', '',
        '## 2. Dataset sample and split', '',
        table([{'item': key, 'value': value} for key, value in setup.items()], [('item', 'Item'), ('value', 'Value')]),
        'Full input/target ID presence, uniqueness, ID sets and exact row alignment were verified before fitting.',
        'Predictor numerical values are loaded only for the sampled rows; the entire ID sequence is checked.',
        'Stratified subsampling uses proportional largest-remainder allocation; the sample is then split',
        f'with class-wise shuffling and approximately {VALIDATION_FRACTION:.0%} validation rows.',
        'The identical sample, row order, split and transformed representation are used for every method.',
        'Input file hashes and split-index hashes are recorded in `run_metadata.json`.', '',
        '## 3. Minimal preprocessing', '',
        f'Id is excluded. All {setup["predictor_columns"]} other columns are retained as numeric baseline inputs.',
        'Only actual blank/NaN/NA values are missing. Codes such as 7, 9, 77, 88 and 99 remain observed.',
        'Per-column medians are fit using training rows only; all-missing training columns use a documented',
        'zero fallback. Training medians are applied to both splits. Means and population standard deviations',
        'are fit on the imputed training matrix only, then reused for validation.',
        f'Standard deviations <= {ZERO_STD_TOLERANCE:g} use scale 1 and that preserved column is zeroed in both splits.',
        f'Training zero-variance columns: **{len(constants)}** ({", ".join(constants) or "none"}).',
        f'All-missing training columns: **{len(all_missing)}** ({", ".join(all_missing) or "none"}).',
        'A leading column of ones supplies an intercept. Course ridge/logistic L2 penalizes every coefficient,',
        'including that intercept; no custom intercept exemption is introduced.',
        '**This crude numeric treatment of survey categories is a baseline representation, not a recommended',
        'final representation. No exact feature-removal or semantic-code decisions are made here.**',
        'See `preprocessing_summary.txt` and `preprocessing_parameters.csv` for the complete training-only statistics.', '',
        '## 4. Majority-class baseline', '',
        table([rows[0]], [('accuracy', 'Accuracy'), ('balanced_accuracy', 'Balanced accuracy'),
                          ('precision', 'Precision'), ('recall', 'Recall'), ('specificity', 'Specificity'), ('f1', 'F1')]),
        'This always-negative reference is not counted as a mandatory method. Undefined precision and any',
        'zero-denominator ratio are reported as 0. High majority accuracy alone is not useful positive detection.', '',
        '## 5. Six mandatory implementations', '',
        '- MSE GD, MSE SGD, least squares and ridge receive `y_pm1` in {-1,+1}; predict +1 when linear score >= 0.',
        '- Logistic methods receive explicit `y_01 = (y_pm1 + 1) / 2`; predict +1 when sigmoid(score) >= 0.5.',
        '- All validation metrics use the original {-1,+1} labels. There is no threshold search.',
        '- Returned training losses are audited at returned weights. They exclude L2 penalties. Raw MSE',
        '  and logistic loss are different objectives and must not be compared numerically across families.',
        '- All six required functions are called unchanged. Singular solves and numerical errors are recorded,',
        '  not replaced by a different solver or silently hidden. Failed rows have blank validation metrics.', '',
        '### Configurations and conditioning', '',
        f'Full-batch budgets: {BATCH_ITERATIONS}; size-one SGD budgets: {SGD_ITERATIONS}.',
        f'Ridge lambdas: {RIDGE_LAMBDAS}; regularized logistic lambdas: {LOGISTIC_LAMBDAS}.',
        'Let L be the largest eigenvalue of training tx.T@tx / n. GD gamma = 1/L;',
        'logistic gamma = 1/(0.25 L); regularized logistic gamma = 1/(0.25 L + 2 lambda).',
        'These smoothness-based steps use only training data and remain fixed within each call.',
        'SGD gamma = 1/max_i ||tx_i||^2, a conservative training-only per-example stability bound.',
        'This can be slow with extreme scaled rows; it is not tuned for validation performance.', '',
        table([{'item': key, 'value': value} for key, value in conditioning.items()], [('item', 'Diagnostic'), ('value', 'Value')]),
        'Gram numerical rank counts eigenvalues above the recorded Gram-specific tolerance; it is a',
        'conditioning diagnostic. Retaining constant/duplicate information can make unregularized normal',
        'equations singular. Such failure is an input-conditioning observation, not a correctness bug in the API.', '',
        f'Attempted mandatory configurations: **{len(rows) - 1}**; successful: **{len(successful)}**;',
        f'failed: **{len(failed)}**. Failures:', '',
        table(failed, [('model', 'Method'), ('config_id', 'Run'), ('lambda', 'Lambda'), ('error_message', 'Failure')]),
        '## 6. Validation results', '',
        'Best observed configuration per successful method, selected by validation balanced accuracy',
        '(then F1/recall and configuration ID for ties). All attempts remain in `model_metrics.csv`.',
        'This exploratory selection on one split is not an unbiased final performance estimate.', '',
        table(winners, numeric_columns),
        '![Metric comparison](figures/01_model_metric_comparison.png)', '',
        '## 7. GD vs SGD vs closed-form least squares', '',
        table([row for row in rows if row['model'] in {'mean_squared_error_gd', 'mean_squared_error_sgd', 'least_squares'}], numeric_columns),
        'These are optimizers/solvers for the same unregularized linear-score family, not three unrelated',
        'predictive families. Full-batch updates visit all training rows; one SGD update visits one row.',
        '`training_data_passes` makes that difference explicit. SGD budgets need not reach a full pass.',
        'Different optimization budgets and step sizes affect loss and classification performance.',
        'Compare finite status, returned data loss and wall-clock time above; a failed direct solve has no',
        'validation comparison. More iterations or a tuned SGD step could change these diagnostic results.', '',
        *optimization_observations, '',
        '### Change at the largest tested iterative budget', '',
        'Each change is relative to the preceding successful restarted run with the same model/lambda.',
        'A material-change flag means absolute relative objective change >=1% or absolute balanced-accuracy',
        'change >=0.01. This is a descriptive heuristic, not an optimizer stopping criterion or proof of convergence.', '',
        table(final_budgets, [('model', 'Method'), ('lambda', 'Lambda'), ('max_iters', 'Updates'),
                              ('training_data_passes', 'Data passes'), ('data_loss_change', 'Data loss change'),
                              ('relative_objective_change', 'Relative objective change'),
                              ('balanced_accuracy_change', 'Bal. acc. change'),
                              ('materially_changing_flag', 'Material change')]),
        '## 8. Effect of ridge regularization', '',
        regularization_comparison(rows, 'least_squares', 'ridge_regression'), '',
        table([row for row in rows if row['model'] == 'ridge_regression'], numeric_columns),
        'With the course scaling, ridge solves tx.T@tx + 2*n*lambda*I. A positive lambda addresses',
        'rank deficiency in this representation. Lambda=0 is the unregularized normal equation.', '',
        '## 9. Logistic vs regularized logistic regression', '',
        regularization_comparison(rows, 'logistic_regression', 'reg_logistic_regression'), '',
        table([row for row in rows if row['model'] in LOGISTIC_METHODS], numeric_columns),
        'Regularized optimization uses the course gradient penalty 2*lambda*w. Regularization also',
        'shrinks the intercept in the required implementation, which can affect class proportions.',
        'Minority-metric improvements at large lambda may partly reflect these score/calibration shifts;',
        'this diagnostic does not isolate intercept shrinkage from changes to the predictor coefficients.',
        'Separate restarted budgets provide convergence evidence without altering mandatory function APIs.', '',
        '![Regularization comparison](figures/05_regularization_comparison.png)', '',
        '## 10. Regression-based vs logistic classification', '',
        family_answer, '',
        '## 11. Class-imbalance behavior', '',
        table([rows[0]] + winners, [('model', 'Method'), ('accuracy', 'Accuracy'), ('actual_positive_fraction', 'Actual positive'),
                                  ('predicted_positive_fraction', 'Predicted positive'), ('recall', 'Recall'),
                                  ('specificity', 'Specificity'), ('f1', 'F1')]),
        f'**{len(collapsed)} successful configurations predict fewer than {COLLAPSE_POSITIVE_FRACTION:.0%} positives**',
        f'and are explicitly flagged in the metrics. Near-all-positive uses >{1 - COLLAPSE_POSITIVE_FRACTION:.0%};',
        'constant-score flag uses score standard deviation <= 1e-12. Low predicted-positive fractions and',
        'low recall paired with high specificity indicate majority-dominated behavior at these fixed thresholds.',
        'Prediction ranges/probabilities are saved in `predictions_summary.csv`; no per-person predictions are exported.', '',
        '![Recall and specificity](figures/02_recall_specificity_comparison.png)',
        '![Predicted positive rates](figures/03_predicted_positive_rate.png)', '',
        '## 12. Computational cost', '',
        f'Input/sample loading: {metadata["loading_runtime_seconds"]:.3f}s; preprocessing and training-only',
        f'conditioning: {metadata["preprocessing_runtime_seconds"]:.3f}s.',
        'Per-model training and prediction timings use perf_counter and exclude input loading/preprocessing.',
        'The training timer wraps the mandatory function, including its final returned loss calculation;',
        'external loss audits and metric calculations are excluded. Prediction timing covers matrix scoring',
        'and class conversion. Direct-solver failures still have elapsed attempt time.',
        'Repeated validation would multiply costs. Compare equal budgets and convergence before extrapolating;',
        'runtime varies with hardware/BLAS and is not deterministic.', '',
        '![Training runtimes](figures/04_runtime_comparison.png)', '',
        '## 13. Important observations for later preprocessing', '',
        f'- The preserved numeric representation has {len(constants)} training constant columns and Gram rank',
        f'  {conditioning["training_gram_numerical_rank"]}/{conditioning["training_gram_dimension"]}; inspect',
        '  numerical redundancy together with the separate EDA before final representation decisions.',
        f'- {len(collapsed)} majority-dominated configurations demonstrate why accuracy alone is inadequate.',
        '- Imputation and scaling are train-only, but numeric survey labels still impose artificial ordering.',
        '  These diagnostics cannot select the final encoding, identify refusal codes, or establish leakage.',
        '- Compare the iteration trajectories in convergence_summary.csv before interpreting weak metrics',
        '  as evidence against a model family. Loss/objective changes and weight norms reveal incomplete fitting.',
        '- The full EDA and this report must be interpreted together; neither is a final feature-processing decision.', '',
        '## 14. Implications for model selection', '',
        f'Best mandatory run by balanced accuracy: {describe_run(best_run(rows, set(METHODS)))}.',
        f'Best mandatory run by F1: {describe_run(best_run(rows, set(METHODS), "f1"))}.',
        'The observed ranking describes only this crude representation, small configuration set and fixed',
        'thresholds. It does not supersede the separate EDA strategy recommendations.',
        'For the next phase, combine semantic/code review with fold-fitted preprocessing, then compare',
        'classification families under identical stratified validation splits and monitored convergence.',
        'Track balanced accuracy, F1, precision, recall and specificity; reserve threshold/cost tuning and',
        'feature engineering for that later phase. Use a fresh holdout or nested validation for final claims.', '',
    ]
    (output / 'baseline_report.md').write_text('\n'.join(report), encoding='utf-8')


def main(argv=None):
    """CLI: run the six mandatory methods on a stratified sample and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=PROJECT_DIR / 'dataset')
    parser.add_argument('--output-dir', type=Path, default=SCRIPT_DIR / 'baseline_diagnostics')
    parser.add_argument('--max-rows', type=int, default=MAX_DIAGNOSTIC_ROWS)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    for protected in [args.data_dir.resolve(), SCRIPT_DIR / 'eda_outputs', IMPLEMENTATIONS_PATH]:
        protected = protected.resolve()
        if output == protected or output in protected.parents or protected in output.parents:
            parser.error('Output directory must be separate from input data, mandatory implementations and EDA outputs')
    if args.max_rows < 4:
        parser.error('--max-rows must be >= 4')
    implementations_path = IMPLEMENTATIONS_PATH
    original_implementation_hash = sha256_file(implementations_path)
    start = time.perf_counter()
    print('Reading targets and drawing the stratified diagnostic sample...', flush=True)
    ids, full_y_pm1 = load_targets(args.data_dir / 'y_train.csv')
    sample_indices = stratified_subsample(full_y_pm1, args.max_rows)
    names, sample, integrity = load_sampled_predictors(args.data_dir / 'x_train.csv', ids, sample_indices)
    y_pm1 = full_y_pm1[sample_indices]
    training_indices, validation_indices = stratified_split(y_pm1)
    loading_runtime = time.perf_counter() - start
    start = time.perf_counter()
    fitted, preprocessing_records = fit_preprocessing(sample[training_indices], names)
    tx_train = transform_predictors(sample[training_indices], fitted)
    tx_validation = transform_predictors(sample[validation_indices], fitted)
    del sample
    y_train_pm1, y_validation_pm1 = y_pm1[training_indices], y_pm1[validation_indices]
    configurations, conditioning = choose_configurations(tx_train)
    preprocessing_runtime = time.perf_counter() - start
    print(f'IDs verified; {len(y_pm1):,} sampled rows, {len(tx_train):,} training, {len(tx_validation):,} validation.', flush=True)
    print(f'Calling all six mandatory methods across {len(configurations)} small diagnostic configurations...', flush=True)
    rows = [majority_baseline(y_train_pm1, y_validation_pm1, tx_train.shape[1])]
    predictions = []
    for config in configurations:
        print(f'  {config["config_id"]}: {config["model"]}, iterations={config["max_iters"]}, lambda={config["lambda"]}', flush=True)
        row, prediction = run_configuration(config, tx_train, tx_validation, y_train_pm1, y_validation_pm1)
        rows.append(row)
        predictions.append(prediction)
    if sha256_file(implementations_path) != original_implementation_hash:
        raise RuntimeError('Mandatory implementations changed during the diagnostic; results not saved')
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'model_metrics.csv', rows)
    write_csv(output / 'predictions_summary.csv', predictions)
    write_csv(output / 'preprocessing_parameters.csv', preprocessing_records)
    convergence = summarize_convergence(rows)
    write_csv(output / 'convergence_summary.csv', convergence)
    metadata = {
        'setup': {
            'source_rows': len(full_y_pm1), 'configured_max_rows': args.max_rows,
            'diagnostic_rows': len(y_pm1), 'predictor_columns': len(names),
            'parameters_including_intercept': tx_train.shape[1],
            'training_rows': len(tx_train), 'validation_rows': len(tx_validation),
            'source_positive_rate': float(np.mean(full_y_pm1 == 1)),
            'sample_positive_rate': float(np.mean(y_pm1 == 1)),
            'training_positive_rate': float(np.mean(y_train_pm1 == 1)),
            'validation_positive_rate': float(np.mean(y_validation_pm1 == 1)),
        },
        'integrity': integrity, 'conditioning': conditioning,
        'seed': RANDOM_SEED, 'split_seed': RANDOM_SEED + 1, 'sgd_seed': RANDOM_SEED + 1001,
        'validation_fraction': VALIDATION_FRACTION, 'batch_iterations': BATCH_ITERATIONS,
        'sgd_iterations': SGD_ITERATIONS, 'ridge_lambdas': RIDGE_LAMBDAS,
        'logistic_lambdas': LOGISTIC_LAMBDAS, 'configurations': configurations,
        'loading_runtime_seconds': loading_runtime, 'preprocessing_runtime_seconds': preprocessing_runtime,
        'python': platform.python_version(), 'numpy': np.__version__,
        'source_sha256': {name: sha256_file(args.data_dir / name) for name in ['x_train.csv', 'y_train.csv']},
        'implementation_sha256_before_and_after': original_implementation_hash,
        'sample_source_indices_sha256': hashlib.sha256(sample_indices.astype('<i8').tobytes()).hexdigest(),
        'training_source_indices_sha256': hashlib.sha256(sample_indices[training_indices].astype('<i8').tobytes()).hexdigest(),
        'validation_source_indices_sha256': hashlib.sha256(sample_indices[validation_indices].astype('<i8').tobytes()).hexdigest(),
    }
    preprocessing_text = (
        'DIAGNOSTIC BASELINE ONLY: integer survey codes are temporarily numeric. No final feature decisions.\n'
        f'Source rows: {len(full_y_pm1)}\nDiagnostic sample: {len(y_pm1)}\n'
        f'Training rows: {len(tx_train)}\nValidation rows: {len(tx_validation)}\n'
        f'Seed: {RANDOM_SEED}; split seed: {RANDOM_SEED + 1}; SGD seed: {RANDOM_SEED + 1001}\n'
        f'Validation fraction: {VALIDATION_FRACTION}\n'
        'Dropped column: Id only. All other input columns are retained.\n'
        'Actual blanks/NaN/NA: median imputation fit on training rows only.\n'
        'All-missing training feature fallback: zero.\n'
        'Scaling: training-imputed population mean and std only; reused for validation.\n'
        f'Zero-variance tolerance: {ZERO_STD_TOLERANCE}; applied scale 1 and column zeroed in both splits.\n'
        f'All-missing training features: {sum(r["all_missing_training_flag"] for r in preprocessing_records)}\n'
        f'Zero-variance training features: {sum(r["zero_variance_training_flag"] for r in preprocessing_records)}\n'
        'Intercept: leading ones column, regularized by required ridge / regularized-logistic functions.\n'
        'MSE/least-squares/ridge target: y_pm1 in {-1,+1}; threshold score >= 0.\n'
        'Logistic target: y_01 = (y_pm1 + 1)/2; threshold probability >= 0.5.\n'
        'Special numeric survey codes remain unchanged; no one-hot encoding or engineered interactions.\n'
        'Training statistics: preprocessing_parameters.csv. Every model receives the identical transformed data.\n'
    )
    (output / 'preprocessing_summary.txt').write_text(preprocessing_text, encoding='utf-8')
    generate_figures(output, rows)
    import matplotlib
    metadata['matplotlib'] = matplotlib.__version__
    (output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    write_report(output, rows, metadata, preprocessing_records)
    successful = [row for row in rows if row['status'] == 'ok']
    fastest = min(successful, key=lambda r: r['training_runtime_seconds']) if successful else None
    print(f'\nMandatory baseline diagnostics complete.\nDiagnostic rows: {len(y_pm1):,}\n'
          f'Training rows: {len(tx_train):,}\nValidation rows: {len(tx_validation):,}\n'
          f'Positive prevalence: {np.mean(y_pm1 == 1):.2%}\n'
          f'Majority baseline balanced accuracy: {rows[0]["balanced_accuracy"]:.3f}\n'
          f'Best mandatory model by balanced accuracy: {describe_run(best_run(rows, set(METHODS)))}\n'
          f'Best mandatory model by F1: {describe_run(best_run(rows, set(METHODS), "f1"))}\n'
          f'Best recall: {describe_run(best_run(rows, set(METHODS), "recall"))}\n'
          f'Fastest successful configuration: {fastest["config_id"] + " " + fastest["model"] if fastest else "none"}\n'
          f'Failed mandatory configurations: {sum(row["status"] == "failed" for row in rows)}\n'
          f'Outputs: {output / "baseline_report.md"}\n'
          f'         {output / "model_metrics.csv"}\n'
          f'         {output / "convergence_summary.csv"}')


if __name__ == '__main__':
    main()
