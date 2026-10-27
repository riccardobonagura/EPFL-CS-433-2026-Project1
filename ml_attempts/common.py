"""Shared augmented fitting, metric definitions and transparent experiment logs."""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from feature_processing.io_utils import PROJECT_DIR
from .plots import chart
from .representations import save_representation

sys.path.insert(0, str(PROJECT_DIR / 'Mandatory Functions'))
from implementations_augmented import (
    classification_metrics, find_best_threshold, fit_mse_sgd, fit_reg_logistic_newton,
    fit_reg_logistic_regression, linear_predict_scores, logistic_predict_proba,
)


PREVIOUS_BENCHMARK = {'representation': 'A0 minimal, previous 100000-row diagnostic',
                      'balanced_accuracy': 0.7671, 'f1': 0.358, 'threshold': 0.1135,
                      'interpretation': 'Historical reference only; threshold is never reused for a new fit'}
METRICS = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'specificity', 'f1', 'tp', 'tn', 'fp', 'fn',
           'actual_positive_fraction', 'predicted_positive_fraction']


def write_json(path, values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(values, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows, keys=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = keys or list(dict.fromkeys(key for row in rows for key in row)) or ['status']
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: '' if value is None or isinstance(value, (float, np.floating)) and not np.isfinite(value)
                             else value for key, value in row.items()})


def metric_values(y_pm1, scores, threshold):
    predicted = np.where(scores >= threshold, 1, -1)
    return {**classification_metrics(y_pm1, predicted),
            'actual_positive_fraction': float(np.mean(y_pm1 == 1)),
            'predicted_positive_fraction': float(np.mean(predicted == 1))}


def predict_scores(model, tx, w):
    """Linear scores for MSE; probabilities for logistic and Newton models."""
    if model == 'mse':
        return linear_predict_scores(tx, w)
    return logistic_predict_proba(tx, w)


def fit_newton(y_pm1, tx, params, *, tx_val=None, y_val_pm1=None, history=False):
    """Full-batch Newton L2 logistic; runs to the gradient tolerance, no early stopping."""
    y_val = None if y_val_pm1 is None else ((y_val_pm1 + 1) / 2).astype(int)
    return fit_reg_logistic_newton(((y_pm1 + 1) / 2).astype(int), tx, params['lambda_'],
                                   np.zeros(tx.shape[1]), params['max_iters'],
                                   gradient_tol=params['gradient_tol'], penalize_intercept=False,
                                   tx_val=tx_val, y_val=y_val, return_history=history)


def fit_model(bundle, params, *, history=True):
    if params['model'] == 'newton':
        start = time.perf_counter()
        result = fit_newton(bundle['train_y'], bundle['train'], params, tx_val=bundle['validation'],
                            y_val_pm1=bundle['validation_y'], history=history)
        return result, time.perf_counter() - start
    options = {key: value for key, value in params.items() if key not in {'model', 'lambda_'}}
    options.update(initial_w=np.zeros(bundle['train'].shape[1]),
                   tx_val=bundle['validation'], return_history=history)
    start = time.perf_counter()
    if params['model'] == 'mse':
        options['y_val'] = bundle['validation_y']
        result = fit_mse_sgd(bundle['train_y'], bundle['train'], **options)
    else:
        options['y_val'] = ((bundle['validation_y'] + 1) / 2).astype(int)
        result = fit_reg_logistic_regression(((bundle['train_y'] + 1) / 2).astype(int), bundle['train'],
                                              lambda_=params['lambda_'], penalize_intercept=False,
                                              sample_weight=None, **options)
    seconds = time.perf_counter() - start
    return result, seconds


SELECTION_METRICS = ('balanced_accuracy', 'f1')


def selection_metric(config):
    """An attempt ranks runs and picks thresholds by BA (default) or by F1."""
    metric = config.get('selection_metric', 'balanced_accuracy')
    if metric not in SELECTION_METRICS:
        raise ValueError('selection_metric must be one of ' + ', '.join(SELECTION_METRICS))
    return metric


def rank_key(row, metric='balanced_accuracy'):
    if row['status'] != 'ok':
        return (-1, -1, -1, -1)
    order = [metric] + [key for key in ['balanced_accuracy', 'f1', 'recall', 'precision'] if key != metric]
    return tuple(row[key] for key in order) + (-row['actual_updates'],)


def best(rows, metric='balanced_accuracy'):
    good = [row for row in rows if row['status'] == 'ok']
    if not good:
        raise RuntimeError('No finite successful configuration in this search stage')
    return max(good, key=lambda row: rank_key(row, metric))


def new_log(context, attempt, config):
    output = context['root'] / 'ml_attempts' / attempt / 'outputs'
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'run_config.txt', {'configuration': config, 'split': context['metadata'],
                                         'previous_benchmark': PREVIOUS_BENCHMARK,
                                         'raw_data_directory': str(context['data_dir'])})
    return {'context': context, 'attempt': attempt, 'output': output, 'config': config, 'rows': [],
            'params': {}, 'recipes': {}, 'thresholds': []}


def run_candidate(log, bundle, recipe, params, stage):
    run_id = f'run_{len(log["rows"]) + 1:03d}'
    folder = log['output'] / 'runs' / run_id
    folder.mkdir(parents=True, exist_ok=True)
    row = {'attempt': log['attempt'], 'run_id': run_id, 'representation': recipe['name'],
           'stage': stage, 'model': params['model'], 'status': 'failed', 'error': '',
           'feature_count': bundle['train'].shape[1] - 1, 'matrix_memory_mb': bundle['matrix_memory_mb'],
           'lambda': params.get('lambda_'), 'gamma': params['gamma'], 'batch_size': params['batch_size'],
           'lr_schedule': params.get('lr_schedule', ''), 'lr_decay': params.get('lr_decay'),
           'max_updates': params['max_iters'], 'actual_updates': 0, 'best_iteration': None,
           'best_validation_loss': None, 'stop_reason': '', 'threshold': None,
           'training_runtime_seconds': 0.0, 'preprocessing_runtime_seconds': bundle['preprocessing_seconds'],
           'train_loss': None, 'weight_norm': None,
           'training_data_passes': None, 'recovered_sources': ';'.join(bundle['state']['recovered_features']),
           **{key: None for key in METRICS}, **{'default_' + key: None for key in METRICS}}
    print(f'  {run_id} {recipe["name"]}: {stage}, lambda={params.get("lambda_")}, '
          f'gamma={params["gamma"]:.4g}, batch={params["batch_size"]}, max={params["max_iters"]}', flush=True)
    start = time.perf_counter()
    try:
        result, seconds = fit_model(bundle, params)
        row.update(training_runtime_seconds=seconds, actual_updates=result.n_updates,
                   best_iteration=result.best_iteration, best_validation_loss=result.best_validation_loss,
                   stop_reason=result.stop_reason, train_loss=result.loss,
                   training_data_passes=result.n_updates * min(params['batch_size'] or len(bundle['train']),
                                                               len(bundle['train'])) / len(bundle['train']))
        if result.stop_reason == 'non_finite' or not np.isfinite(result.w).all():
            raise FloatingPointError('Augmented fitter detected nonfinite optimization')
        scores = predict_scores(params['model'], bundle['validation'], result.w)
        before = result.w.copy()
        threshold, _ = find_best_threshold(bundle['validation_y'], scores, metric=selection_metric(log['config']))
        np.testing.assert_array_equal(before, result.w)
        default_threshold = 0.0 if params['model'] == 'mse' else 0.5
        selected = metric_values(bundle['validation_y'], scores, threshold)
        default = metric_values(bundle['validation_y'], scores, default_threshold)
        row.update(selected)
        row.update({'default_' + key: value for key, value in default.items()})
        row.update(status='ok', threshold=threshold, weight_norm=float(np.linalg.norm(result.w)))
        np.savez_compressed(folder / 'weights.npz', w=result.w)
        np.save(folder / 'validation_scores.npy', scores, allow_pickle=False)
        if result.history:
            write_csv(folder / 'history.csv', [dict(zip(result.history, values)) for values in zip(*result.history.values())])
        for policy, value, metrics in [('default', default_threshold, default), ('validation_selected', threshold, selected)]:
            log['thresholds'].append({'run_id': run_id, 'representation': recipe['name'], 'policy': policy,
                                      'threshold': value, **metrics})
        print(f'    BA default={default["balanced_accuracy"]:.4f}, selected={selected["balanced_accuracy"]:.4f}, '
              f'F1={selected["f1"]:.4f}; {result.n_updates} updates / {result.stop_reason}', flush=True)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        row['error'] = f'{type(error).__name__}: {error}'
        row['training_runtime_seconds'] = max(row['training_runtime_seconds'], time.perf_counter() - start)
        print('    FAILED: ' + row['error'], flush=True)
    log['rows'].append(row)
    log['params'][run_id] = dict(params)
    log['recipes'][run_id] = recipe
    write_json(folder / 'configuration.json', {'recipe': recipe, 'model_params': params, 'split_sha256': log['context']['metadata']['split_sha256']})
    for filename in ['all_runs.csv', 'validation_metrics.csv']:
        write_csv(log['output'] / filename, log['rows'])
    write_csv(log['output'] / 'threshold_results.csv', log['thresholds'])
    return row


def select(log, row):
    payload = {'attempt': log['attempt'], 'best_run': row, 'model_params': log['params'][row['run_id']],
               'recipe': log['recipes'][row['run_id']], 'split': log['context']['metadata'],
               'previous_benchmark': PREVIOUS_BENCHMARK, 'selection_metric': selection_metric(log['config']),
               'confirmation_complete': False}
    write_json(log['output'] / 'selection.json', payload)
    scores = np.load(log['output'] / 'runs' / row['run_id'] / 'validation_scores.npy')
    values = np.unique(np.r_[np.quantile(scores, np.linspace(0, 1, 41)), row['threshold'], 0 if row['model'] == 'mse' else 0.5])
    curve = [{'threshold': float(value), **metric_values(log['context']['y_pm1'][log['context']['validation']], scores, value)} for value in values]
    write_csv(log['output'] / 'selected_threshold_curve.csv', curve)
    chart(log['output'] / 'figures/threshold_vs_metrics.png', 'Validation threshold tradeoff: selected model',
          [r['threshold'] for r in curve], {key: [r[key] for r in curve] for key in ['balanced_accuracy', 'f1', 'recall']},
          'Probability threshold' if row['model'] != 'mse' else 'Linear score threshold', 'Validation metric (fraction)', unit_interval=True)
    history_path = log['output'] / 'runs' / row['run_id'] / 'history.csv'
    if history_path.exists():
        with history_path.open(newline='', encoding='utf-8') as stream:
            history = list(csv.DictReader(stream))
        chart(log['output'] / 'figures/convergence.png', 'Selected run: unregularized data-loss checkpoints',
              [float(r['update']) for r in history],
              {key: [float(r[key]) for r in history] for key in ['train_loss', 'validation_loss']},
              'Gradient updates', 'Data loss (MSE and logistic losses are not comparable)')
    return payload


def report_header(title, log):
    meta = log['context']['metadata']
    return [f'# {title}', '',
            f'Shared split: **{meta["training_rows"]:,} training / {meta["validation_rows"]:,} validation rows**;',
            f'original source rows {meta["source_rows"]:,}. Split hash `{meta["split_sha256"]}`.',
            'Sampling/split seeds 42/43. Raw model inputs are read only from dataset/. No test rows were read during selection.',
            'Existing preprocessing, augmented fitters, metrics, schedules and threshold helpers are reused.',
            'All starts are zero; there are no class weights and no intercept penalty for L2 logistic fits.', '',
            'Historical 100,000-row preprocessing benchmark: A0 BA≈0.7671, F1≈0.358, threshold≈0.1135.',
            'This threshold is contextual only, never assigned to a newly trained model. Current full-split results',
            'are not a paired comparison with that smaller diagnostic.', '',
            '| Run | Representation | Lambda | Gamma | Batch | Updates | Default BA | Selected BA | F1 | Recall |',
            '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']


def report_rows(rows):
    return [f'| {r["run_id"]} | {r["representation"]} | {r["lambda"]} | {r["gamma"]:.4g} | {r["batch_size"]} | '
            f'{r["actual_updates"]} | {r["default_balanced_accuracy"]:.4f} | {r["balanced_accuracy"]:.4f} | {r["f1"]:.4f} | {r["recall"]:.4f} |'
            for r in rows if r['status'] == 'ok']


def report_footer(log):
    failed = [r for r in log['rows'] if r['status'] != 'ok']
    return ['', f'Failed/nonfinite runs: {len(failed)}; details remain in all_runs.csv.', '',
            '## Selection limits and completion', '',
            'Early stopping, thresholds and configuration comparison reuse the exploratory validation set.',
            'These scores are selection diagnostics, not untouched final estimates. Differences at a tuned',
            'threshold partly reflect calibration; compare default-threshold metrics alongside them.',
            'A max_iters stop is not convergence. Independent budgets and batch sizes need not represent equal work.',
            'Finalist confirmation uses 3 stratified outer folds with an inner stopping/threshold partition;',
            'preprocessing is refitted on each inner-training partition and outer labels never select thresholds.',
            'The finalist recipes were chosen using overlapping exploratory data, so confirmation remains',
            'conditional on that prior selection and is not a new clinical/generalization guarantee.',
            'Confirmation results and final all-training refits are recorded below when complete.', '']
