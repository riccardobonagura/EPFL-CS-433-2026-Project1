"""Fold-isolated finalist confirmation, then frozen all-training refits/inference."""

import csv
import hashlib
import json
import math
import time

import numpy as np

from feature_processing.io_utils import load_unlabeled_csv, stratified_split
from .common import (
    METRICS, classification_metrics, find_best_threshold, fit_model, fit_mse_sgd,
    fit_reg_logistic_regression, linear_predict_scores, logistic_predict_proba,
    metric_values, write_csv, write_json,
)
from .data import stratified_folds, validate_sources
from .representations import fit_representation, prepare, save_representation, transform_representation


def _selected_updates(payload):
    row = payload['best_run']
    return row['actual_updates'] if row['best_iteration'] is None else row['best_iteration']


def confirm_finalist(context, payload):
    """Outer labels are evaluated only after inner stopping/threshold selection."""
    output = context['root'] / 'ml_attempts' / payload['attempt'] / 'outputs'
    folder = output / 'confirmation'
    folder.mkdir(parents=True, exist_ok=True)
    pool = np.sort(np.r_[context['train'], context['validation']])
    folds = stratified_folds(context['y_pm1'][pool], n_folds=3, seed=2026)
    rows, protocol = [], hashlib.sha256()
    for fold_number, relative in enumerate(folds, 1):
        held_out = pool[relative]
        remaining = np.ones(len(pool), dtype=bool)
        remaining[relative] = False
        outer_training = pool[remaining]
        inner_training, stopping = stratified_split(context['y_pm1'][outer_training], validation_fraction=0.20, seed=900 + fold_number)
        training_indices, stopping_indices = outer_training[inner_training], outer_training[stopping]
        assert not np.intersect1d(training_indices, held_out).size
        assert not np.intersect1d(stopping_indices, held_out).size
        for values in [training_indices, stopping_indices, held_out]:
            protocol.update(values.astype('<i8').tobytes())
        print(f'Confirming {payload["attempt"]}: outer fold {fold_number}/3...', flush=True)
        bundle = prepare(context, payload['recipe'], training_indices, stopping_indices)
        params = dict(payload['model_params'])
        scale = len(training_indices) / payload['best_run']['n_train'] if 'n_train' in payload['best_run'] else len(training_indices) / context['metadata']['training_rows']
        params['max_iters'] = max(1, int(math.ceil(_selected_updates(payload) * scale)))
        params['eval_every'] = max(1, min(params['eval_every'], max(1, params['max_iters'] // 8)))
        if params['lr_schedule'] == 'inverse':
            params['lr_decay'] /= scale
        result, seconds = fit_model(bundle, params)
        if result.stop_reason == 'non_finite':
            raise RuntimeError('Nonfinite finalist confirmation: ' + payload['attempt'])
        stopping_scores = (linear_predict_scores(bundle['validation'], result.w) if params['model'] == 'mse'
                           else logistic_predict_proba(bundle['validation'], result.w))
        threshold, _ = find_best_threshold(bundle['validation_y'], stopping_scores)
        outer_matrix = transform_representation(context['x'][held_out], context['names'], bundle['state'])
        outer_scores = (linear_predict_scores(outer_matrix, result.w) if params['model'] == 'mse'
                        else logistic_predict_proba(outer_matrix, result.w))
        selected_metrics = metric_values(context['y_pm1'][held_out], outer_scores, threshold)
        default_metrics = metric_values(context['y_pm1'][held_out], outer_scores, 0 if params['model'] == 'mse' else 0.5)
        selected_iteration = result.best_iteration if result.best_iteration is not None else result.n_updates
        row = {'fold': fold_number, 'inner_training_rows': len(training_indices),
               'inner_stopping_rows': len(stopping_indices), 'outer_validation_rows': len(held_out),
               'threshold': threshold, 'max_updates': params['max_iters'], 'actual_updates': result.n_updates,
               'best_iteration': result.best_iteration, 'stop_reason': result.stop_reason,
               'training_runtime_seconds': seconds, 'feature_count': outer_matrix.shape[1] - 1,
               'selected_training_passes': selected_iteration * min(params['batch_size'], len(training_indices)) / len(training_indices),
               **selected_metrics, **{'default_' + key: value for key, value in default_metrics.items()}}
        rows.append(row)
        fold_output = folder / f'fold_{fold_number}'
        save_representation(bundle['state'], fold_output / 'preprocessing_state')
        write_json(fold_output / 'model_config.json', params)
        np.savez_compressed(fold_output / 'indices.npz', training=training_indices, stopping=stopping_indices, validation=held_out)
        np.savez_compressed(fold_output / 'weights.npz', w=result.w, threshold=threshold)
        if result.history:
            write_csv(fold_output / 'history.csv', [dict(zip(result.history, values)) for values in zip(*result.history.values())])
        write_csv(output / 'cv_metrics.csv', rows)
        print(f'  Outer BA={selected_metrics["balanced_accuracy"]:.4f}, F1={selected_metrics["f1"]:.4f}; '
              f'inner-only threshold={threshold:.5g}', flush=True)
        del bundle, outer_matrix
    summary = {}
    for metric in ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'specificity', 'f1']:
        values = [row[metric] for row in rows]
        summary['mean_' + metric] = float(np.mean(values))
        summary['std_' + metric] = float(np.std(values, ddof=1))
    summary.update(threshold=float(np.median([row['threshold'] for row in rows])),
                   median_training_passes=float(np.median([row['selected_training_passes'] for row in rows])),
                   mean_inner_training_rows=float(np.mean([row['inner_training_rows'] for row in rows])),
                   protocol_sha256=protocol.hexdigest(), outer_folds=3,
                   protocol='Stratified outer folds; 20% inner stopping/threshold split; fit only inner training; no outer refit')
    payload.update(confirmation_complete=True, confirmation=summary)
    write_json(output / 'selection.json', payload)
    report_path = output / 'experiment_report.md'
    report = report_path.read_text(encoding='utf-8').split('\n## Finalist confirmation', 1)[0]
    report += '\n## Finalist confirmation\n\n'
    report += '| Fold | Outer BA | Outer F1 | Recall | Precision | Inner threshold |\n| --- | --- | --- | --- | --- | --- |\n'
    report += '\n'.join(f'| {row["fold"]} | {row["balanced_accuracy"]:.4f} | {row["f1"]:.4f} | '
                        f'{row["recall"]:.4f} | {row["precision"]:.4f} | {row["threshold"]:.5g} |' for row in rows)
    report += (f'\n\nMean outer BA **{summary["mean_balanced_accuracy"]:.3%}**, fold SD **{summary["std_balanced_accuracy"]:.3%}**. '
               f'Mean F1 **{summary["mean_f1"]:.3%}**.\n\n'
               'Each outer fold independently fits preprocessing and OOF target encoding on inner-training rows only. '
               'The inner stopping partition selects stopping and thresholds. Outer labels never select either. '
               'Models deliberately use about 53% of the confirmation pool per fold (two-thirds times 80%); there is no outer-training '
               'refit after calibration. This conservative protocol is identical for all finalists. '
               'Prior exploratory configuration selection overlapped these source rows, so this is conditional confirmation, '
               'not a fully nested audit of the complete configuration search.\n\n'
               'Final inference threshold is the median of the three inner-selected thresholds, fixed before any test-file access. '
               'Final training duration uses median selected data passes; no threshold or duration uses outer performance or test data.\n')
    report_path.write_text(report, encoding='utf-8')
    validate_sources(context)
    return payload


def _fit_full(training, y_pm1, params):
    options = {key: value for key, value in params.items() if key not in {'model', 'lambda_'}}
    options.update(initial_w=np.zeros(training.shape[1]), early_stopping=False, return_history=False)
    start = time.perf_counter()
    if params['model'] == 'mse':
        result = fit_mse_sgd(y_pm1, training, **options)
    else:
        result = fit_reg_logistic_regression(((y_pm1 + 1) / 2).astype(int), training,
                                              lambda_=params['lambda_'], penalize_intercept=False,
                                              sample_weight=None, **options)
    if result.stop_reason == 'non_finite':
        raise RuntimeError('Final all-data model diverged; no submission was exported')
    return result, time.perf_counter() - start


def write_submission(path, ids, labels):
    ids, labels = np.asarray(ids), np.asarray(labels)
    if ids.ndim != 1 or labels.shape != ids.shape or len(set(ids.tolist())) != len(ids):
        raise ValueError('Submission IDs/predictions must be aligned, unique and one-dimensional')
    if not np.isin(labels, [-1, 1]).all():
        raise ValueError('Submission predictions must use {-1,+1}')
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Id', '_MICHD'])
        writer.writerows((record_id, int(label)) for record_id, label in zip(ids, labels))


def final_refit(context, payload):
    if not payload.get('confirmation_complete'):
        raise ValueError('Confirm a finalist before all-data training or test access')
    output = context['root'] / 'ml_attempts' / payload['attempt'] / 'outputs'
    confirmation = payload['confirmation']
    threshold = confirmation['threshold']
    params = dict(payload['model_params'])
    params['early_stopping'] = False
    batch = min(params['batch_size'], len(context['x']))
    params['max_iters'] = int(math.ceil(confirmation['median_training_passes'] * len(context['x']) / batch))
    if params['lr_schedule'] == 'inverse':
        params['lr_decay'] *= context['metadata']['training_rows'] / len(context['x'])
    # Freeze all selected choices before inference-only access to x_test.csv.
    frozen = {'recipe': payload['recipe'], 'params': params, 'threshold': threshold,
              'full_training_rows': len(context['x']), 'stopping_source': 'median inner-CV selected data passes',
              'threshold_source': 'median inner-CV selected threshold', 'test_used_for_selection': False}
    write_json(output / 'frozen_final_configuration.json', frozen)
    print(f'All-data refit {payload["attempt"]}: {len(context["x"]):,} rows, {params["max_iters"]} fixed updates...', flush=True)
    training, state, preprocessing_seconds = fit_representation(context['x'], context['y_pm1'], context['names'], payload['recipe'])
    result, seconds = _fit_full(training, context['y_pm1'], params)
    del training
    save_representation(state, output / 'preprocessing_state')
    np.savez_compressed(output / 'final_model_weights.npz', w=result.w, threshold=threshold,
                        output_features=np.asarray(state['output_features']), labels=np.array([-1, 1]))
    # This is the first point in the workflow that reads test data.
    raw_test, test_names, test_ids = load_unlabeled_csv(context['data_dir'] / 'x_test.csv')
    test_matrix = transform_representation(raw_test, test_names, state)
    linear_scores = linear_predict_scores(test_matrix, result.w)
    probabilities = logistic_predict_proba(test_matrix, result.w) if params['model'] != 'mse' else None
    scores = linear_scores if probabilities is None else probabilities
    labels = np.where(scores >= threshold, 1, -1)
    write_submission(output / 'submission.csv', test_ids, labels)
    with (output / 'test_predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Id', 'score', 'probability', '_MICHD'])
        writer.writerows((record_id, float(score), '' if probabilities is None else float(probability), int(label))
                         for record_id, score, probability, label in zip(test_ids, linear_scores,
                                                                        scores if probabilities is None else probabilities, labels))
    final = {'training_rows': len(context['x']), 'test_rows': len(raw_test), 'n_updates': result.n_updates,
             'train_data_loss': result.loss, 'training_runtime_seconds': seconds,
             'preprocessing_runtime_seconds': preprocessing_seconds, 'threshold': threshold,
             'test_predicted_positive_fraction': float(np.mean(labels == 1)),
             'feature_count': len(result.w) - 1,
             'matrix_memory_mb': len(context['x']) * len(result.w) * 8 / 1024**2,
             'local_export_only': True}
    payload.update(final_refit=final)
    write_json(output / 'selection.json', payload)
    write_json(output / 'final_refit.json', final)
    report_path = output / 'experiment_report.md'
    report = report_path.read_text(encoding='utf-8').split('\n## Final all-data refit', 1)[0]
    report += (f'\n## Final all-data refit\n\nAll **{len(context["x"]):,}** labeled rows fit the final preprocessing state and model. '
               f'The fixed training budget was **{result.n_updates} updates** and threshold **{threshold:.6g}**. '
               f'Fit time: {seconds:.3f}s. **{len(raw_test):,}** test rows were transformed after selection and exported '
               'in original ID order.\n\nEarly stopping was used for selection/confirmation. The all-row refit has no '
               'validation rows, so it uses the previously selected data-pass budget with early stopping disabled; '
               'inverse decay is rescaled to preserve its epoch interpretation. No test labels or leaderboard feedback '
               'are used. final_model_weights.npz and preprocessing_state/ describe the all-row fit; submission.csv '
               'contains only Id,_MICHD with -1/+1 labels. Files are local and were not submitted externally.\n')
    report_path.write_text(report, encoding='utf-8')
    validate_sources(context)
    del raw_test, test_matrix
    return payload


def master_comparison(context):
    rows = []
    for folder in sorted((context['root'] / 'ml_attempts').glob('0*/outputs')):
        path = folder / 'selection.json'
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding='utf-8'))
        if payload['split'] != context['metadata']:
            continue
        row = dict(payload['best_run'])
        row['exploratory_threshold'] = row['threshold']
        if payload.get('confirmation_complete'):
            row.update({'cv_' + key: value for key, value in payload['confirmation'].items() if isinstance(value, (float, int))})
            row['threshold'] = payload['confirmation']['threshold']
            row['cv_protocol_sha256'] = payload['confirmation']['protocol_sha256']
        if payload.get('final_refit'):
            row['final_training_runtime_seconds'] = payload['final_refit']['training_runtime_seconds']
            row['final_updates'] = payload['final_refit']['n_updates']
            row['final_feature_count'] = payload['final_refit']['feature_count']
            row['final_matrix_memory_mb'] = payload['final_refit']['matrix_memory_mb']
        rows.append(row)
    protocols = {row['cv_protocol_sha256'] for row in rows if 'cv_protocol_sha256' in row}
    if len(protocols) > 1:
        raise ValueError('Finalists were not confirmed on identical outer/inner rows')
    write_csv(context['root'] / 'ml_attempts/model_comparison.csv', rows)
    completed = [row for row in rows if 'cv_mean_balanced_accuracy' in row]
    completed.sort(key=lambda row: (-row['cv_mean_balanced_accuracy'], -row['cv_mean_f1']))
    lines = ['# Three Project 1 experiments', '',
             '| Attempt | Representation | Exploratory BA | 3-fold BA mean | Fold SD | 3-fold F1 |',
             '| --- | --- | --- | --- | --- | --- |']
    for row in completed:
        lines.append(f'| {row["attempt"]} | {row["representation"]} | {row["balanced_accuracy"]:.4f} | '
                     f'{row["cv_mean_balanced_accuracy"]:.4f} | {row["cv_std_balanced_accuracy"]:.4f} | {row["cv_mean_f1"]:.4f} |')
    if completed:
        lines += ['', 'Highest observed confirmation mean: **' + completed[0]['attempt'] + '**.',
                  'Only three preselected finalists were confirmed. Small differences and overlapping fold variability',
                  'do not establish a decisive winner; recipes were selected earlier using overlapping exploratory data.',
                  'Each attempt has its own local, all-training refit and submission; nothing was uploaded.', '']
    lines += ['', '## Local artifacts', '']
    for row in rows:
        attempt = row['attempt']
        lines.append(f'- {attempt}: [report]({attempt}/outputs/experiment_report.md), '
                     f'[submission]({attempt}/outputs/submission.csv), '
                     f'[validation metrics]({attempt}/outputs/validation_metrics.csv).')
    (context['root'] / 'ml_attempts/summary.md').write_text('\n'.join(lines), encoding='utf-8')
