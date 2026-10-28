"""Controlled cumulative processing ablations with one fixed augmented model."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'feature_processing'

from .feature_config import get_config
from .io_utils import PROJECT_DIR, check_modules, file_sha256, load_training_partitions, protect_modules, validate_output_directory
from .preprocessing import fit_transform, save_state, transform_with_diagnostics, write_manifest
from .run_preprocessing import build_config

sys.path.insert(0, str(PROJECT_DIR / 'Mandatory Functions'))
from implementations_augmented import classification_metrics, find_best_threshold, fit_reg_logistic_regression, logistic_predict_proba


def ablation_configs(type_hints=None, extra_quarantine=None, include_target=True):
    """Return the ordered (label, config) ablation ladder from A0 minimal to the full profiles."""
    common = {'type_hints': type_hints or {}}
    if extra_quarantine is not None:
        common['quarantine_features'] = extra_quarantine
    configs = [
        ('A0 minimal baseline', get_config('minimal', **common)),
        ('A1 + safe removals', get_config('minimal', safe_removals=True, **common)),
        ('A2 + missingness treatment', get_config('minimal', safe_removals=True, missingness=True, **common)),
        ('A3 + low-cardinality encoding', get_config('curated_v1', canonical=False, **common)),
        ('A4 + canonical representatives', get_config('curated_v1', **common)),
        ('A5 + controlled engineering', get_config('curated_v1_engineered', **common)),
    ]
    if include_target:
        configs.append(('A6 + cross-fitted target encoding', get_config('compact_target_encoded', engineer=True, **common)))
    return configs


def _intercept(x):
    """Prepend a column of ones to x."""
    result = np.empty((len(x), x.shape[1] + 1), dtype=float)
    result[:, 0], result[:, 1:] = 1.0, x
    return result


def write_ablation_report(output, rows, setup, model_config):
    """Write the Markdown report comparing every ablation on the shared split."""
    lines = ['# Processing ablation diagnostics', '',
             f'Sample: {setup["sample_rows"]:,} of {setup["source_rows"]:,} source rows; shared train/validation split hash:',
             f'`{setup["split_sha256"]}`. Sampling seed 42 and split seed 43 match baseline_diagnostics.py.', '',
             'All transformations are fit on the same training rows; validation is transformed with stored state.',
             'One fixed L2 logistic setup is used throughout: no intercept penalty, no class weights,',
             'zero initialization, local seeded mini-batches and validation-based early stopping.',
             'Model configuration: `' + json.dumps(model_config, sort_keys=True) + '`.', '',
             '| Stage | Dimensions | Default BA | Selected BA | Selected F1 | Recall | Precision | Specificity | Threshold | Fit seconds | Matrix GiB |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        lines.append(f'| {row["ablation"]} | {row["n_features"]} | {row["default_balanced_accuracy"]:.4f} | '
                     f'{row["balanced_accuracy"]:.4f} | {row["f1"]:.4f} | {row["recall"]:.4f} | '
                     f'{row["precision"]:.4f} | {row["specificity"]:.4f} | {row["selected_threshold"]:.4g} | '
                     f'{row["training_runtime_seconds"]:.2f} | {row["matrix_memory_gib"]:.3f} |')
    best = max(rows, key=lambda row: (row['balanced_accuracy'], row['f1'], row['recall']))
    lines += ['', f'Best exploratory selected-threshold result: **{best["ablation"]}**, balanced accuracy',
              f'**{best["balanced_accuracy"]:.2%}**, F1 **{best["f1"]:.2%}**. This is not an untouched estimate.', '',
              '## Cumulative changes', '',
              'All stages exclude Id/SEQNO/_PSU and the same default quarantine. A0 is consequently an',
              'approximation to the previous crude baseline, not an exact reproduction of its predictor set.',
              'A1 adds training-observed constants and configured exact-duplicate removal. A2 adds sparse raw',
              'removal, deduplicated presence/missing indicators and scaled global missing_count.',
              'A3 jointly introduces verified code decoding, verified binary mapping, categorical handling and',
              'exclusion of unverified numeric/unsupported medium-cardinality values. It is a representation-policy',
              'bundle; its delta cannot be attributed to one-hot encoding alone. A4 selects explicit concept profiles.',
              'A5 adds only available validated square/interaction recipes. A6 builds on A5 with OOF target encoding.', '',
              '| Change | Selected BA difference | Default BA difference |', '| --- | --- | --- |']
    for previous, current in zip(rows, rows[1:]):
        lines.append(f'| {previous["ablation"][:2]} → {current["ablation"][:2]} | '
                     f'{current["balanced_accuracy"] - previous["balanced_accuracy"]:+.5f} | '
                     f'{current["default_balanced_accuracy"] - previous["default_balanced_accuracy"]:+.5f} |')
    lines += ['', '## Interpretation limits', '',
              'Default-threshold and selected-threshold metrics are both recorded; threshold selection changes no',
              'weights. Differences at selected thresholds can partly reflect calibration rather than representation.',
              'The validation set selects both stopping and thresholds; comparing several representations adds',
              'selection optimism. Confirm finalists with stratified folds and a fresh holdout before final claims.',
              'Identical update budgets need not mean equal convergence across dimensions. Histories and actual',
              'update counts are saved so underfitting can be inspected. No model hyperparameter search is performed.',
              'Default curated policy is conservative about semantic uncertainty; a dropped feature is not proven',
              'uninformative. Codebook review and additional explicit semantic overrides are separate experiments.', '',
              '![Ablation metrics](figures/01_ablation_metrics.png)',
              '![Representation dimensions](figures/02_ablation_dimensions.png)', '',
              'State, manifest and compact convergence history for each stage are under `ablations/A0` ... `A6`.', '']
    (output / 'ablation_report.md').write_text('\n'.join(lines), encoding='utf-8')


def figures(output, rows):
    """Save matplotlib bar charts of the ablation metrics to output/figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = output / 'figures'
    folder.mkdir(exist_ok=True)
    indices = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, key, label in [(-0.25, 'default_balanced_accuracy', 'BA at threshold 0.5'),
                               (0, 'balanced_accuracy', 'BA at selected threshold'),
                               (0.25, 'f1', 'F1 at selected threshold')]:
        ax.bar(indices + offset, [row[key] for row in rows], width=0.25, label=label)
    ax.set_xticks(indices, [row['ablation'][:2] for row in rows])
    ax.set(title='Cumulative preprocessing ablations on one validation split', xlabel='Processing stage', ylabel='Validation metric (fraction)', ylim=(0, 1))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2)
    fig.tight_layout()
    fig.savefig(folder / '01_ablation_metrics.png', dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(indices, [row['n_features'] for row in rows], color='#3b6c8d')
    ax.set_xticks(indices, [row['ablation'][:2] for row in rows])
    ax.set(title='Dense representation size by processing stage', xlabel='Processing stage', ylabel='Output dimensions (intercept excluded)')
    fig.tight_layout()
    fig.savefig(folder / '02_ablation_dimensions.png', dpi=140)
    plt.close(fig)


def main(argv=None):
    """CLI: fit each ablation, train one fixed L2 logistic setup and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=PROJECT_DIR / 'dataset')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'outputs')
    parser.add_argument('--max-rows', type=int, default=100_000)
    parser.add_argument('--max-updates', type=int, default=1500)
    parser.add_argument('--skip-target-encoding', action='store_true')
    parser.add_argument('--eda-summary', type=Path, default=PROJECT_DIR / 'Exploratory Analysis/eda_outputs/feature_summary.csv')
    parser.add_argument('--semantic-review', type=Path)
    args = parser.parse_args(argv)
    if args.max_updates < 1:
        parser.error('--max-updates must be >=1')
    protected = protect_modules()
    output = validate_output_directory(args.output_dir, args.data_dir)
    fingerprints = {name: file_sha256(args.data_dir / name) for name in ['x_train.csv', 'y_train.csv']}
    print('Loading one shared stratified processing-evaluation sample...', flush=True)
    data = load_training_partitions(args.data_dir, args.max_rows)
    shared = build_config('curated_v1', args.eda_summary, args.semantic_review)
    model_config = {'lambda_': 0.01, 'gamma': 0.03, 'batch_size': 1024, 'max_iters': args.max_updates,
                    'seed': 42, 'early_stopping': True, 'patience': 5, 'tol': 1e-6,
                    'eval_every': 50, 'penalize_intercept': False}
    rows = []
    for label, config in ablation_configs(shared['type_hints'], shared['quarantine_features'], not args.skip_target_encoding):
        print('Evaluating ' + label + '...', flush=True)
        start = time.perf_counter()
        training, state = fit_transform(data['x_train'], data['feature_names'], y=data['y_train'], config=config)
        validation, diagnostics = transform_with_diagnostics(data['x_validation'], data['feature_names'], state)
        processing_seconds = time.perf_counter() - start
        matrix_memory = (training.nbytes + validation.nbytes) / 1024**3
        tx_train, tx_validation = _intercept(training), _intercept(validation)
        start = time.perf_counter()
        fit = fit_reg_logistic_regression((data['y_train'] + 1) / 2, tx_train,
                                          initial_w=np.zeros(tx_train.shape[1]), sample_weight=None,
                                          tx_val=tx_validation, y_val=(data['y_validation'] + 1) / 2,
                                          return_history=True, **model_config)
        training_seconds = time.perf_counter() - start
        if fit.stop_reason == 'non_finite':
            raise RuntimeError('Nonfinite evaluator fit for ' + label)
        probability = logistic_predict_proba(tx_validation, fit.w)
        y_validation = (data['y_validation'] + 1) / 2
        default = classification_metrics(y_validation, (probability >= 0.5).astype(int))
        threshold, metrics = find_best_threshold(y_validation, probability)
        row = {'ablation': label, 'n_train': len(training), 'n_validation': len(validation),
               'n_features': training.shape[1], 'matrix_memory_gib': matrix_memory,
               'processing_runtime_seconds': processing_seconds, 'training_runtime_seconds': training_seconds,
               'selected_threshold': threshold, 'n_updates': fit.n_updates, 'stop_reason': fit.stop_reason,
               'best_iteration': fit.best_iteration, 'best_validation_loss': fit.best_validation_loss,
               'train_loss': fit.loss, 'unknown_validation_categories': diagnostics['unknown_category_total'],
               **metrics, **{'default_' + key: value for key, value in default.items()}}
        rows.append(row)
        folder = output / 'ablations' / label[:2]
        folder.mkdir(parents=True, exist_ok=True)
        save_state(state, folder / 'preprocessing_state.json')
        write_manifest(state, folder / 'feature_manifest.csv')
        with (folder / 'convergence_history.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(fit.history)
            writer.writerows(zip(*fit.history.values()))
        print(f'  {training.shape[1]} dimensions; default BA={default["balanced_accuracy"]:.4f}; '
              f'selected BA={metrics["balanced_accuracy"]:.4f}; F1={metrics["f1"]:.4f}', flush=True)
        del training, validation, tx_train, tx_validation
    with (output / 'ablation_results.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {'source_rows': data['source_rows'], 'sample_rows': data['sample_rows'],
                'split_sha256': data['split_sha256'], 'model_config': model_config,
                'source_sha256': fingerprints, 'protected_ml_modules': protected}
    (output / 'ablation_run.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    figures(output, rows)
    write_ablation_report(output, rows, metadata, model_config)
    check_modules(protected)
    assert fingerprints == {name: file_sha256(args.data_dir / name) for name in fingerprints}
    best = max(rows, key=lambda row: (row['balanced_accuracy'], row['f1'], row['recall']))
    print(f'\nProcessing ablations complete. Best exploratory stage: {best["ablation"]}; '
          f'BA={best["balanced_accuracy"]:.2%}, F1={best["f1"]:.2%}.\nReport: {output / "ablation_report.md"}')


if __name__ == '__main__':
    main()
