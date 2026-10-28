"""Compose existing preprocessing and OOF encoders; no new encoding algorithms."""

import json
import time
from pathlib import Path

import numpy as np

from feature_processing import fit_preprocessor, fit_transform, load_state, save_state, transform, write_manifest
from feature_processing.evaluate_processing import ablation_configs
from feature_processing.target_encoding import cross_fit_target_encode, transform_target_encoder


def base_recipes():
    """The three Attempt 1 representations: R0 minimal, R1 missingness, R2 curated."""
    profiles = ablation_configs(include_target=False)
    return {name: {'name': name, 'kind': 'standard', 'config': profiles[index][1]}
            for name, index in [('R0 minimal', 0), ('R1 missingness', 2), ('R2 curated', 4)]}


def _recoverable(raw, names, options):
    """Pick the dropped medium-cardinality fields eligible for target encoding (TE2)."""
    config = base_recipes()['R2 curated']['config']
    state = fit_preprocessor(raw, names, config=config)
    selected = []
    for name in options['candidate_priority']:
        if name not in state.specs or name in config['quarantine_features']:
            continue
        spec = state.specs[name]
        if spec.get('drop_reason') != 'unused_medium_cardinality':
            continue
        values = raw[:, names.index(name)]
        observed = values[np.isfinite(values)]
        _, counts = np.unique(observed, return_counts=True)
        minimum = max(config['rare_min_count'], int(np.ceil(config['rare_min_fraction'] * len(raw))))
        if len(observed) >= options['minimum_observed'] and np.sum(counts >= minimum) >= 2:
            selected.append(name)
        if len(selected) == options['max_recovered_features']:
            break
    return selected


def _append_intercept(base, extra=None):
    """Stack [1, base, extra] column-wise into one float matrix."""
    extra_width = 0 if extra is None else extra.shape[1]
    result = np.empty((len(base), base.shape[1] + extra_width + 1), dtype=float)
    result[:, 0], result[:, 1:base.shape[1] + 1] = 1.0, base
    if extra_width:
        result[:, base.shape[1] + 1:] = extra
    return result


def fit_representation(raw, y_pm1, names, recipe):
    """Training uses core fit_transform, including OOF columns where requested."""
    start = time.perf_counter()
    config = recipe['config']
    base, base_state = fit_transform(raw, names, y=y_pm1, config=config)
    extra, encoder, selected = None, None, []
    if recipe['kind'] == 'recovery':
        selected = _recoverable(raw, names, recipe['recovery'])
        if selected:
            extra, encoder = cross_fit_target_encode(
                raw[:, [names.index(name) for name in selected]], ((y_pm1 + 1) / 2).astype(int), selected,
                n_folds=recipe['recovery']['target_folds'], seed=recipe['recovery']['target_seed'],
                alpha=recipe['alpha'], rare_min_count=config['rare_min_count'],
                rare_min_fraction=config['rare_min_fraction'],
            )
    outputs = ['intercept'] + base_state.output_features + [name + '__recovered_target' for name in selected]
    if len(set(outputs)) != len(outputs):
        raise ValueError('Representation produced duplicate output names')
    result = _append_intercept(base, extra)
    state = {'recipe': recipe, 'base': base_state, 'encoder': encoder,
             'recovered_features': selected, 'output_features': outputs}
    return result, state, time.perf_counter() - start


def transform_representation(raw, names, state):
    """Apply a fitted representation to new raw rows (intercept included)."""
    base = transform(raw, names, state['base'])
    extra = None
    if state['encoder'] is not None:
        selected = state['recovered_features']
        extra = transform_target_encoder(raw[:, [names.index(name) for name in selected]], state['encoder'], selected)
    return _append_intercept(base, extra)


def prepare(context, recipe, train_indices=None, validation_indices=None):
    """Fit a recipe on training indices and transform validation; return the matrices and state."""
    train_indices = context['train'] if train_indices is None else train_indices
    validation_indices = context['validation'] if validation_indices is None else validation_indices
    raw = context['x'][train_indices]
    training, state, seconds = fit_representation(raw, context['y_pm1'][train_indices], context['names'], recipe)
    del raw
    start = time.perf_counter()
    validation = transform_representation(context['x'][validation_indices], context['names'], state)
    seconds += time.perf_counter() - start
    return {'train': training, 'validation': validation, 'state': state,
            'train_y': context['y_pm1'][train_indices], 'validation_y': context['y_pm1'][validation_indices],
            'preprocessing_seconds': seconds,
            'matrix_memory_mb': (training.nbytes + validation.nbytes) / 1024**2}


def save_representation(state, folder):
    """Save a fitted representation (state JSON, manifest, metadata) to a folder."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    save_state(state['base'], folder / 'preprocessing_state.json')
    write_manifest(state['base'], folder / 'feature_manifest.csv')
    metadata = {key: value for key, value in state.items() if key != 'base'}
    (folder / 'representation.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load_representation(folder):
    """Restore a complete standard or recovery representation for inference."""
    folder = Path(folder)
    state = json.loads((folder / 'representation.json').read_text(encoding='utf-8'))
    state['base'] = load_state(folder / 'preprocessing_state.json')
    return state
