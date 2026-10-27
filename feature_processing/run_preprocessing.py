"""Fit a reusable preprocessing state on one deterministic training partition."""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'feature_processing'

from .feature_config import CODEBOOK_URL, get_config, load_eda_hypotheses, load_semantic_review
from .io_utils import PROJECT_DIR, check_modules, file_sha256, load_training_partitions, load_unlabeled_csv, protect_modules, validate_output_directory
from .preprocessing import fit_transform, load_state, save_state, transform, transform_with_diagnostics, write_manifest


def build_config(profile, eda_path=None, review_path=None):
    config = get_config(profile)
    if eda_path and Path(eda_path).is_file():
        config['type_hints'] = load_eda_hypotheses(eda_path)
    if review_path:
        config['quarantine_features'] = sorted(set(config['quarantine_features']) | set(load_semantic_review(review_path)))
    return config


def write_processing_report(output, state, partitions, training, validation, diagnostics, elapsed):
    reasons = Counter(spec.get('drop_reason', '') for spec in state.specs.values() if spec['kind'] == 'dropped')
    count = lambda reason: reasons[reason]
    lines = [
        '# Feature processing report', '',
        f'Profile: **{state.config["profile"]}**. Source: {partitions["source_rows"]:,} rows;',
        f'diagnostic/processing selection: {partitions["sample_rows"]:,}. Seeds: sampling 42, split 43.',
        f'Training: {len(training):,}; validation: {len(validation):,}. All fitted quantities use training only.', '',
        '| Quantity | Value |', '| --- | --- |',
        f'| Raw predictors (Id excluded) | {len(state.input_features)} |',
        f'| Dropped raw features | {len(state.dropped_features)} |',
        f'| Output dimensions | {len(state.output_features)} |',
        f'| Verified binary sources | {len(state.binary_features)} |',
        f'| One-hot categorical sources | {len(state.categorical_features)} |',
        f'| Cross-fitted target-encoded sources | {len(state.target_features)} |',
        f'| Numeric sources | {len(state.numeric_features)} |',
        f'| Separate missing/presence indicators | {len(state.missing_indicator_features)} |',
        f'| Engineered recipes | {len(state.engineered_feature_config.get("recipes", []))} |',
        f'| Constants removed | {count("constant")} |',
        f'| Exact duplicates removed | {count("exact_duplicate")} |',
        f'| Extremely sparse raw values removed | {count("sparse_raw_value")} |',
        f'| Quarantined sources excluded | {count("quarantine")} |',
        f'| Unused medium categorical sources | {count("unused_medium_cardinality")} |',
        f'| Unverified numeric sources excluded | {count("unverified_numeric")} |',
        f'| Dtype | {training.dtype} |',
        f'| Training matrix shape | {training.shape} |',
        f'| Validation matrix shape | {validation.shape} |',
        f'| Matrix memory (train + validation) | {(training.nbytes + validation.nbytes) / 1024**3:.3f} GiB |',
        f'| Unknown validation category occurrences | {diagnostics["unknown_category_total"]} |',
        f'| Processing wall-clock seconds | {elapsed:.3f} |', '',
        '## Policies and interpretation', '',
        'IDs are preserved separately and never enter the model matrix. No intercept is added by preprocessing.',
        'Median imputation, clipping thresholds, scaling, category vocabularies, rare groups and engineering',
        'normalization are fitted on training rows only. Unknown categories use the reserved __OTHER__ level;',
        '__MISSING__ stays separate. Ordinary reference levels are recorded in the state.',
        'Reserved missing/unknown columns can be zero on training rows so future levels remain representable.',
        'The cardinality limit counts retained nonmissing levels plus an observed rare group; missing and',
        'unseen-category reserves are additional. No blanket one-hot expansion or raw numeric treatment of categories.', '',
        'Global missing_count uses actual raw NaNs, before decoding or imputation, from original eligible',
        'predictors. Identifiers, configured exact-duplicate copies and quarantined sources are excluded',
        'from that count to prevent indirect quarantine or duplicate contributions. Its training count is scaled.',
        'No presence flag or engineered term is created from a quarantined source. Complete EDA semantic-review',
        'quarantine is opt-in via --semantic-review; the default list is recorded in preprocessing_state.json.', '',
        'Special codes are decoded only by explicit per-field verified mappings. Current defaults cover',
        'GENHLTH, BPHIGH4, BPMEDS and _RFHYPE5; BPMEDS and _RFHYPE5 also have verified binary maps.',
        f'The [CDC 2015 codebook]({CODEBOOK_URL}) supplies those definitions and numeric age/height/weight/BMI semantics.',
        'Project CSV decimals are used as supplied; no extra implied-decimal conversion is applied.',
        'Unverified numeric candidates are excluded in curated profiles unless explicitly overridden.',
        'Raw versus derived concept choices are explicit configurable profiles, not correlation-based deletion.', '',
        '## Unknown validation categories', '',
        '| Source | Unseen nonmissing occurrences |', '| --- | --- |',
    ]
    lines.extend(f'| {name} | {count} |' for name, count in sorted(diagnostics['unknown_category_counts'].items(), key=lambda item: -item[1]) if count)
    lines.extend(['', '## Reuse and limitations', '',
                  'preprocessing_state.json contains the complete fitted transformation and schema.',
                  'The saved state was reloaded and verified to reproduce validation transforms exactly.',
                  'Target encoding, when enabled, uses OOF training values with fold-specific priors and vocabularies;',
                  'validation/test use only full-training encoder statistics. Use fit_transform for training.',
                  'Validation and test data never select transformations. No classifier is fitted by this runner.',
                  'Use evaluate_processing.py for controlled processing comparisons; its validation thresholds',
                  'and early-stopping scores are selection diagnostics, not untouched final estimates.', '',
                  'Detailed decisions: [feature_manifest.csv](feature_manifest.csv).',
                  'Processing comparisons, once run: [ablation_report.md](ablation_report.md).', ''])
    (output / 'processing_report.md').write_text('\n'.join(lines), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=PROJECT_DIR / 'dataset')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'outputs')
    parser.add_argument('--profile', default='curated_v1')
    parser.add_argument('--max-rows', type=int, default=0, help='0 uses the full training CSV')
    parser.add_argument('--save-arrays', action='store_true')
    parser.add_argument('--test-csv', type=Path)
    parser.add_argument('--eda-summary', type=Path, default=PROJECT_DIR / 'Exploratory Analysis/eda_outputs/feature_summary.csv')
    parser.add_argument('--semantic-review', type=Path)
    args = parser.parse_args(argv)
    protected = protect_modules()
    output = validate_output_directory(args.output_dir, args.data_dir)
    sources = {name: file_sha256(args.data_dir / name) for name in ['x_train.csv', 'y_train.csv']}
    print('Loading training data and making the deterministic stratified split...', flush=True)
    partitions = load_training_partitions(args.data_dir, args.max_rows)
    config = build_config(args.profile, args.eda_summary, args.semantic_review)
    start = time.perf_counter()
    print('Fitting training-only preprocessing state...', flush=True)
    training, state = fit_transform(partitions['x_train'], partitions['feature_names'], y=partitions['y_train'], config=config)
    validation, diagnostics = transform_with_diagnostics(partitions['x_validation'], partitions['feature_names'], state)
    elapsed = time.perf_counter() - start
    save_state(state, output / 'preprocessing_state.json')
    restored = load_state(output / 'preprocessing_state.json')
    # Check a bounded subset to avoid allocating a second full validation matrix.
    np.testing.assert_array_equal(transform(partitions['x_validation'][:100], partitions['feature_names'], restored), validation[:100])
    assert training.shape[1] == validation.shape[1] == len(state.output_features)
    assert training.dtype == np.dtype(config['output_dtype']) and np.isfinite(training).all()
    write_manifest(state, output / 'feature_manifest.csv')
    if args.save_arrays:
        np.savez_compressed(output / 'X_train_processed.npz', X=training)
        np.savez_compressed(output / 'X_validation_processed.npz', X=validation)
        for name in ['y_train', 'y_validation', 'ids_train', 'ids_validation']:
            np.save(output / (name + '.npy'), partitions[name], allow_pickle=False)
    if args.test_csv:
        test_x, test_names, ids_test = load_unlabeled_csv(args.test_csv)
        test_processed = transform(test_x, test_names, state)
        if args.save_arrays:
            np.savez_compressed(output / 'X_test_processed.npz', X=test_processed)
            np.save(output / 'ids_test.npy', ids_test, allow_pickle=False)
        diagnostics['test_shape'] = list(test_processed.shape)
    metadata = {'source_sha256': sources, 'split_sha256': partitions['split_sha256'],
                'integrity': partitions['integrity'], 'validation_diagnostics': diagnostics,
                'protected_ml_modules': protected, 'seed': 42, 'split_seed': 43}
    (output / 'processing_run.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    write_processing_report(output, state, partitions, training, validation, diagnostics, elapsed)
    check_modules(protected)
    assert sources == {name: file_sha256(args.data_dir / name) for name in sources}
    print(f'\nFeature processing complete.\nProfile: {args.profile}\nRaw predictors: {len(state.input_features)}\n'
          f'Dropped: {len(state.dropped_features)}\nBinary mapped: {len(state.binary_features)}\n'
          f'One-hot source features: {len(state.categorical_features)}\nNumeric features: {len(state.numeric_features)}\n'
          f'Missing indicators: {len(state.missing_indicator_features)}\n'
          f'Engineered features: {len(state.engineered_feature_config.get("recipes", []))}\n'
          f'Final dimensions: {training.shape} / {validation.shape}\ndtype: {training.dtype}\n'
          f'Matrix memory: {(training.nbytes + validation.nbytes) / 1024**3:.3f} GiB\n'
          f'Unknown validation categories: {diagnostics["unknown_category_total"]}\n'
          f'State: {output / "preprocessing_state.json"}\nManifest: {output / "feature_manifest.csv"}\n'
          f'Report: {output / "processing_report.md"}')


if __name__ == '__main__':
    main()
