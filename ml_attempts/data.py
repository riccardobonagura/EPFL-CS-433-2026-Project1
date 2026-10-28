"""Dataset-only raw I/O and persisted original-row split indices."""

import hashlib
import json
from pathlib import Path

import numpy as np

from feature_processing.io_utils import (
    PROJECT_DIR, check_modules, file_sha256, load_sampled_predictors,
    load_targets, protect_modules, stratified_split, stratified_subsample,
)


def load_context(root=PROJECT_DIR, *, max_rows=0):
    """Load the training CSVs and the shared seeded 80/20 split, saving or verifying its indices."""
    root = Path(root)
    data_dir = root / 'dataset'
    ids, y = load_targets(data_dir / 'y_train.csv')
    selected = stratified_subsample(y, max_rows or len(y), seed=42)
    train, validation = stratified_split(y[selected], seed=43)
    train_source, validation_source = selected[train], selected[validation]
    # Original CSV row order is retained in memory; stored indices address it.
    names, x, integrity = load_sampled_predictors(data_dir / 'x_train.csv', ids, np.arange(len(y)))
    source_hashes = {name: file_sha256(data_dir / name) for name in ['x_train.csv', 'y_train.csv']}
    split_hash = hashlib.sha256(train_source.astype('<i8').tobytes() + validation_source.astype('<i8').tobytes()).hexdigest()
    shared = root / 'ml_attempts/shared_split'
    if max_rows and max_rows < len(y):
        shared = shared / f'sample_{max_rows}'
    shared.mkdir(parents=True, exist_ok=True)
    metadata = {'source_sha256': source_hashes, 'split_sha256': split_hash,
                'sampling_seed': 42, 'split_seed': 43, 'validation_fraction': 0.2,
                'training_rows': len(train_source), 'validation_rows': len(validation_source),
                'source_rows': len(y), 'sample_rows': len(selected), 'index_space': 'original CSV data rows, zero-based'}
    if (shared / 'metadata.json').exists():
        previous = json.loads((shared / 'metadata.json').read_text(encoding='utf-8'))
        if previous != metadata:
            raise ValueError('Saved shared split belongs to different source data; use a separate output root')
        if not (np.array_equal(np.load(shared / 'train_indices.npy'), train_source)
                and np.array_equal(np.load(shared / 'validation_indices.npy'), validation_source)):
            raise ValueError('Saved shared split indices differ from their deterministic reconstruction')
    else:
        np.save(shared / 'train_indices.npy', train_source, allow_pickle=False)
        np.save(shared / 'validation_indices.npy', validation_source, allow_pickle=False)
        (shared / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return {'root': root, 'data_dir': data_dir, 'x': x, 'y_pm1': y, 'ids': np.asarray(ids),
            'names': names, 'train': train_source, 'validation': validation_source,
            'metadata': metadata, 'integrity': integrity, 'protected': protect_modules()}


def validate_sources(context):
    """Raise if the raw training files or ML implementation files changed during a run."""
    check_modules(context['protected'])
    for name, digest in context['metadata']['source_sha256'].items():
        if file_sha256(context['data_dir'] / name) != digest:
            raise RuntimeError('Raw training data changed during execution: ' + name)


def stratified_folds(y_pm1, n_folds=3, seed=2026):
    """Split row indices into n_folds class-balanced folds with a fixed seed."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_folds)]
    for label in (-1, 1):
        indices = rng.permutation(np.flatnonzero(y_pm1 == label))
        if len(indices) < n_folds:
            raise ValueError('Each class needs at least n_folds rows')
        for fold, part in zip(folds, np.array_split(indices, n_folds)):
            fold.extend(part.tolist())
    return [rng.permutation(np.asarray(fold, dtype=int)) for fold in folds]
