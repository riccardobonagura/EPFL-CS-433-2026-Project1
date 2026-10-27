"""Smoothed, fold-isolated target encoding with explicit missing/other levels."""

import numpy as np


MISSING = '__MISSING__'
OTHER = '__OTHER__'


def category_tokens(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or np.isinf(values).any():
        raise ValueError('Categories must be 1-D numeric values with optional NaN')
    unique, inverse = np.unique(values, return_inverse=True)
    labels = [MISSING if np.isnan(value) else format(0.0 if value == 0 else value, '.17g') for value in unique]
    return np.asarray(labels, dtype=str)[inverse]


def learn_categories(values, min_count, min_fraction):
    tokens = category_tokens(values)
    levels, counts = np.unique(tokens, return_counts=True)
    minimum = max(min_count, int(np.ceil(min_fraction * len(tokens))))
    kept = sorted(level for level, count in zip(levels, counts) if level != MISSING and count >= minimum)
    rare = sorted(level for level, count in zip(levels, counts) if level != MISSING and count < minimum)
    mapping = {level: (OTHER if level in rare else level) for level in levels}
    return tokens, kept, rare, mapping


def _validate(x, y=None, feature_names=None):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or not len(x) or not x.shape[1] or np.isinf(x).any():
        raise ValueError('Target-encoding data must have nonempty rows/features and no infinities')
    names = list(feature_names) if feature_names is not None else [f'feature_{i}' for i in range(x.shape[1])]
    if len(names) != x.shape[1] or len(set(names)) != len(names):
        raise ValueError('Target-encoding feature names must match unique input columns')
    if y is not None:
        y = np.asarray(y, dtype=float)
        if y.shape != (len(x),) or not np.isin(y, [0, 1]).all():
            raise ValueError('Target encoding requires 1-D {0,1} labels')
    return x, y, names


def fit_target_encoder(x, y, feature_names=None, *, alpha=20.0, rare_min_count=100, rare_min_fraction=0.001):
    x, y, names = _validate(x, y, feature_names)
    if not np.isfinite(alpha) or alpha < 0 or rare_min_count < 1 or not 0 <= rare_min_fraction <= 1:
        raise ValueError('Invalid target smoothing or category-support settings')
    state = {'feature_names': names, 'prior': float(y.mean()), 'alpha': float(alpha),
             'rare_min_count': rare_min_count, 'rare_min_fraction': rare_min_fraction, 'features': {}}
    for index, name in enumerate(names):
        tokens, kept, rare, mapping = learn_categories(x[:, index], rare_min_count, rare_min_fraction)
        mapped = np.array([mapping[token] for token in tokens])
        levels, inverse, counts = np.unique(mapped, return_inverse=True, return_counts=True)
        totals = np.bincount(inverse, weights=y)
        rates = (totals + alpha * state['prior']) / (counts + alpha)
        state['features'][name] = {'values': dict(zip(levels.tolist(), rates.tolist())),
                                   'kept': kept, 'rare': rare, 'category_map': mapping}
    return state


def transform_target_encoder(x, state, feature_names=None):
    x, _, names = _validate(x, feature_names=feature_names or state['feature_names'])
    if set(names) != set(state['feature_names']):
        raise ValueError('Target-encoding columns do not match fitted state')
    positions = {name: index for index, name in enumerate(names)}
    result = np.empty((len(x), len(state['feature_names'])))
    for index, name in enumerate(state['feature_names']):
        tokens = category_tokens(x[:, positions[name]])
        spec = state['features'][name]
        result[:, index] = [spec['values'].get(spec['category_map'].get(token, MISSING if token == MISSING else OTHER),
                                              state['prior']) for token in tokens]
    return result


def cross_fit_target_encode(x, y, feature_names=None, *, n_folds=5, seed=42, **options):
    """Return (OOF training matrix, full-training encoder for unseen rows).

    Fold membership depends only on row positions and a local RNG, not labels.
    Each fold learns its own prior, category vocabulary, rare groups and rates.
    """
    x, y, names = _validate(x, y, feature_names)
    if not isinstance(n_folds, int) or not 2 <= n_folds <= len(x):
        raise ValueError('n_folds must be an integer between 2 and N')
    result = np.empty_like(x, dtype=float)
    folds = np.array_split(np.random.default_rng(seed).permutation(len(x)), n_folds)
    for held_out in folds:
        train_mask = np.ones(len(x), dtype=bool)
        train_mask[held_out] = False
        encoder = fit_target_encoder(x[train_mask], y[train_mask], names, **options)
        result[held_out] = transform_target_encoder(x[held_out], encoder, names)
    return result, fit_target_encoder(x, y, names, **options)
