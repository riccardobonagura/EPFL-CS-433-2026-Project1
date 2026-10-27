"""Training-only fitted preprocessing state and deterministic unseen-data transforms."""

import csv
import hashlib
import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .feature_config import IDENTIFIERS, get_config
from .feature_engineering import apply_recipe, fit_engineering
from .target_encoding import MISSING, OTHER, category_tokens, cross_fit_target_encode, learn_categories, transform_target_encoder


@dataclass
class PreprocessingState:
    input_features: list
    output_features: list = field(default_factory=list)
    dropped_features: list = field(default_factory=list)
    binary_features: list = field(default_factory=list)
    categorical_features: list = field(default_factory=list)
    numeric_features: list = field(default_factory=list)
    medians: dict = field(default_factory=dict)
    means: dict = field(default_factory=dict)
    stds: dict = field(default_factory=dict)
    category_maps: dict = field(default_factory=dict)
    rare_category_maps: dict = field(default_factory=dict)
    missing_indicator_features: list = field(default_factory=list)
    special_code_map: dict = field(default_factory=dict)
    engineered_feature_config: dict = field(default_factory=dict)
    dtype: str = 'float32'
    specs: dict = field(default_factory=dict)
    manifest: list = field(default_factory=list)
    config: dict = field(default_factory=dict)
    target_encoder: dict = field(default_factory=dict)
    target_features: list = field(default_factory=list)
    missing_count_sources: list = field(default_factory=list)
    training_signature: str = ''
    training_rows: int = 0
    notes: list = field(default_factory=list)
    version: int = 1


def _validate(x, feature_names):
    x = np.asarray(x)
    names = list(feature_names)
    if x.ndim != 2 or x.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError('x must be (N,D), with one unique feature name per column')
    if not names or not all(isinstance(name, str) for name in names):
        raise ValueError('Feature names must be nonempty unique strings')
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError('Raw data must be numeric; use NaN for missing cells')
    if np.isinf(x).any():
        raise ValueError('Infinite raw values are not supported')
    return x, names


def _key(value):
    return format(0.0 if float(value) == 0 else float(value), '.17g')


def _normalize_codes(mapping):
    result = {}
    for name, actions in mapping.items():
        result[name] = {}
        for code, action in actions.items():
            if not np.isfinite(float(code)):
                raise ValueError('Special-code keys must be finite')
            if isinstance(action, str):
                if action not in ('missing', 'category'):
                    raise ValueError('Special-code action must be missing, category or numeric')
            elif not np.isfinite(float(action)):
                raise ValueError('Special-code replacements must be finite')
            else:
                action = float(action)
            result[name][_key(code)] = action
    return result


def _decode(values, actions):
    result = np.asarray(values, dtype=float).copy()
    for code, action in actions.items():
        selected = values == float(code)  # Simultaneous mappings, not chained replacements.
        if action == 'missing':
            result[selected] = np.nan
        elif action != 'category':
            result[selected] = float(action)
    return result


def _signature(x, names):
    digest = hashlib.sha256()
    for name in sorted(names):
        values = np.asarray(x[:, names.index(name)], dtype='<f8').copy()
        values[np.isnan(values)] = np.nan
        values[values == 0] = 0
        digest.update(name.encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _manifest(state, source, transformation, *, output='', reason='', dropped=False, stats=None, group=''):
    stats = stats or {}
    state.manifest.append({
        'output_feature': output, 'source_feature': source, 'transformation': transformation,
        'feature_group': group, 'kept_or_dropped': 'dropped' if dropped else 'kept',
        'reason': reason or transformation, 'drop_reason': reason if dropped else '',
        'training_missing_fraction': stats.get('missing_fraction'),
        'training_unique_count': stats.get('unique_count'),
    })


def _add_output(state, output, source, transformation, stats=None, group=''):
    if output in state.output_features:
        raise ValueError('Duplicate output feature name: ' + output)
    state.output_features.append(output)
    _manifest(state, source, transformation, output=output, stats=stats, group=group)


def _add_indicator(state, source, values, kind, registry, stats, group):
    indicator = np.isfinite(values) if kind == 'present' else np.isnan(values)
    output = source + '__' + kind
    digest = np.packbits(indicator).tobytes()
    if digest in registry:
        _manifest(state, source, 'indicator', output=output, reason='duplicate_indicator_of:' + registry[digest],
                  dropped=True, stats=stats, group=group)
        return
    registry[digest] = output
    state.specs.setdefault(source, {}).setdefault('indicators', []).append(kind)
    state.missing_indicator_features.append(output)
    _add_output(state, output, source, kind, stats, group)


def _prepare_numeric(values, name, config, spec=None):
    result = values.copy()
    if spec is None:
        spec = {'log1p': name in config['log1p_features']}
        if name in config['clip_features']:
            low, high = config['clip_features'][name]
            if not 0 <= low <= high <= 100:
                raise ValueError('Clipping percentiles must satisfy 0 <= low <= high <= 100')
            observed = result[np.isfinite(result)]
            spec['clip'] = np.percentile(observed, [low, high]).tolist() if len(observed) else [0.0, 0.0]
    if spec.get('log1p') and np.any(result[np.isfinite(result)] < 0):
        raise ValueError('log1p requires nonnegative decoded values: ' + name)
    if 'clip' in spec:
        result = np.clip(result, *spec['clip'])
    if spec.get('log1p'):
        result = np.log1p(result)
    return result, spec


def _fit_numeric(state, name, values, config, stats, group):
    values, numeric_spec = _prepare_numeric(values, name, config)
    observed = values[np.isfinite(values)]
    median = float(np.median(observed)) if len(observed) else 0.0
    imputed = np.where(np.isnan(values), median, values)
    mean, std = float(imputed.mean()), float(imputed.std())
    state.medians[name], state.means[name], state.stds[name] = median, mean, max(std, 1e-12)
    output = name + '__scaled'
    state.specs[name].update(kind='numeric', numeric=numeric_spec, output=output, zero_variance=std <= 1e-12)
    state.numeric_features.append(name)
    _add_output(state, output, name, 'median_impute_and_scale', stats, group)
    return np.zeros(len(values)) if std <= 1e-12 else (imputed - mean) / std


def _canonical_exclusions(config):
    excluded, groups = set(), {}
    for group, members in config['redundancy_groups'].items():
        groups.update({name: group for name in members})
        if not config['canonical'] or group not in config['group_choices']:
            continue
        choice = config['group_choices'][group]
        profiles = config['group_profiles'].get(group, {})
        if choice not in profiles:
            raise ValueError(f'Unknown {group} representation: {choice}')
        excluded.update(set(members) - set(profiles[choice]))
    return excluded, groups


def fit_preprocessor(x, feature_names, *, y=None, config=None):
    """Learn training-only state. For target-encoded training use fit_transform."""
    x, names = _validate(x, feature_names)
    if not len(x):
        raise ValueError('Cannot fit on zero rows')
    options = dict(config or {})
    config = get_config(options.pop('profile', 'curated_v1'), **options)
    state = PreprocessingState(names, config=config, dtype=config['output_dtype'], training_rows=len(x))
    state.special_code_map = _normalize_codes(config['special_code_map'])
    # The minimal reference retains survey codes, as the original diagnostics did.
    if not config['encode_categorical']:
        state.special_code_map = {}
    quarantine = set() if config['include_quarantined'] else set(config['quarantine_features'])
    exclusions, groups = _canonical_exclusions(config)
    positions = {name: index for index, name in enumerate(names)}
    state.missing_count_sources = [name for name in names if name not in set(IDENTIFIERS) | quarantine
                                   and name not in config['exact_duplicates']]
    indicators, engineering_values, source_outputs = {}, {}, {}
    for name in names:
        values = _decode(x[:, positions[name]], state.special_code_map.get(name, {}))
        observed = values[np.isfinite(values)]
        stats = {'missing_fraction': float(np.isnan(values).mean()), 'unique_count': len(np.unique(observed))}
        group = groups.get(name, '')
        state.specs[name] = {'kind': 'dropped', 'stats': stats, 'group': group}
        reason = None
        if name in IDENTIFIERS:
            reason = 'identifier'
        elif name in quarantine:
            reason = 'quarantine'
        elif config['safe_removals'] and name in config['exact_duplicates']:
            representative = config['exact_duplicates'][name]
            if representative in positions and np.array_equal(x[:, positions[name]], x[:, positions[representative]], equal_nan=True):
                reason = 'exact_duplicate'
                state.specs[name]['representative'] = representative
            elif representative in positions:
                state.notes.append(f'{name} differs from declared duplicate {representative}; not removed as exact duplicate')
        if reason is None and config['safe_removals'] and stats['unique_count'] <= 1:
            reason = 'constant'
        if reason is None and config['missingness'] and stats['missing_fraction'] > config['sparse_threshold']:
            reason = 'sparse_raw_value'
        if reason is None and name in exclusions:
            reason = 'redundant_representation'
        if reason:
            state.dropped_features.append(name)
            state.specs[name]['drop_reason'] = reason
            _manifest(state, name, 'drop_raw', reason=reason, dropped=True, stats=stats, group=group)
            # Duplicate raw constant fields can still contribute a unique presence
            # pattern, but only if the representative is itself constant/sparse.
            eligible_presence = reason in {'constant', 'sparse_raw_value'}
            if reason == 'exact_duplicate':
                representative_values = x[:, positions[state.specs[name]['representative']]]
                observed_rep = representative_values[np.isfinite(representative_values)]
                eligible_presence = (len(np.unique(observed_rep)) <= 1 or
                                     np.isnan(representative_values).mean() > config['sparse_threshold'])
            if (config['missingness'] and config['preserve_presence'] and eligible_presence and
                    0 < stats['missing_fraction'] < 1):
                _add_indicator(state, name, values, 'present', indicators, stats, group)
            continue
        if not config['encode_categorical']:
            _fit_numeric(state, name, values, config, stats, group)
            if config['missingness'] and stats['missing_fraction'] >= config['missing_indicator_threshold']:
                _add_indicator(state, name, values, 'missing', indicators, stats, group)
            continue
        binary_map = config['binary_maps'].get(name)
        forced_category = any(action == 'category' for action in state.special_code_map.get(name, {}).values())
        semantic = config['semantic_types'].get(name)
        if semantic == 'binary' and binary_map is None:
            raise ValueError('Binary semantic overrides require a verified binary map: ' + name)
        hint = config['type_hints'].get(name, '')
        if binary_map is not None and not forced_category:
            mapping = {_key(value): float(result) for value, result in binary_map.items()}
            if set(mapping.values()) != {0.0, 1.0}:
                raise ValueError('Verified binary maps must describe both 0 and 1 states')
            tokens = category_tokens(values)
            mapped = np.array([mapping.get(token, np.nan) for token in tokens])
            state.specs[name].update(kind='binary', mapping=mapping, output=name + '__binary')
            state.binary_features.append(name)
            _add_output(state, name + '__binary', name, 'verified_binary_map_zero_fill', stats, group)
            _add_indicator(state, name, values, 'missing', indicators, stats, group)
            state.specs[name]['other_output'] = name + '__' + OTHER
            _add_output(state, name + '__' + OTHER, name, 'binary_unknown_category', stats, group)
            engineering_values[name] = np.nan_to_num(mapped, nan=0.0)
            source_outputs[name] = name + '__binary'
            continue
        numeric = semantic == 'numeric' or (config['allow_inferred_numeric'] and
                                            ('continuous' in hint or not np.all(observed == np.floor(observed))))
        if numeric and not forced_category:
            result = _fit_numeric(state, name, values, config, stats, group)
            engineering_values[name], source_outputs[name] = result, name + '__scaled'
            if config['missingness'] and (stats['missing_fraction'] >= config['missing_indicator_threshold'] or
                                         name in config['explicit_missing_indicators']):
                _add_indicator(state, name, values, 'missing', indicators, stats, group)
            continue
        categorical = (forced_category or semantic == 'categorical' or 'categorical' in hint or hint == 'binary'
                       or stats['unique_count'] <= config['max_target_levels'])
        if not categorical:
            reason = 'unverified_numeric'
        else:
            tokens, kept, rare, mapping = learn_categories(values, config['rare_min_count'], config['rare_min_fraction'])
            effective = len(kept) + bool(rare)
            limit = config['onehot_limits'].get(name, config['max_onehot_levels'])
            if name == '_AGEG5YR' and config['group_choices'].get('age') == 'age_group_onehot':
                limit = max(limit, 20)  # Explicit opt-in to the age-group representation.
            if effective <= limit:
                reference = kept[0] if kept else None
                levels = [level for level in kept if level != reference] + [MISSING, OTHER]
                state.specs[name].update(kind='categorical', levels=levels, reference=reference,
                                         outputs=[name + '__' + level for level in levels])
                state.category_maps[name], state.rare_category_maps[name] = mapping, rare
                state.categorical_features.append(name)
                for level in levels:
                    _add_output(state, name + '__' + level, name, 'reference_onehot:' + level, stats, group)
                if name in config['explicit_missing_indicators']:
                    _add_indicator(state, name, values, 'missing', indicators, stats, group)
                continue
            selected = config['target_features'] is None or name in config['target_features']
            if config['target_encoding'] and selected and stats['unique_count'] <= config['max_target_levels']:
                state.specs[name].update(kind='target_encoded', output=name + '__target')
                state.target_features.append(name)
                _add_output(state, name + '__target', name, 'cross_fitted_target_encoding', stats, group)
                continue
            reason = 'unused_medium_cardinality'
        state.dropped_features.append(name)
        state.specs[name]['drop_reason'] = reason
        _manifest(state, name, 'drop_raw', reason=reason, dropped=True, stats=stats, group=group)

    if state.target_features:
        if y is None:
            raise ValueError('The compact target-encoded profile requires training targets')
        target_x = np.column_stack([_decode(x[:, positions[name]], state.special_code_map.get(name, {}))
                                    for name in state.target_features])
        oof, encoder = cross_fit_target_encode(target_x, _target_01(y, len(x)), state.target_features,
                                               **_target_options(config))
        state.target_encoder = encoder
        state.training_signature = _signature(x, names)
        for index, name in enumerate(state.target_features):
            # Keep probability-scale TE outputs unscaled. Normalizing all OOF
            # values together could transmit a row's target through other folds'
            # encodings into its own mean/std-adjusted feature.
            state.means[name], state.stds[name] = 0.0, 1.0
            state.category_maps[name] = encoder['features'][name]['category_map']
            state.rare_category_maps[name] = encoder['features'][name]['rare']
    if config['missingness']:
        count = sum(np.isnan(x[:, positions[name]]) for name in state.missing_count_sources)
        count = np.asarray(count, dtype=float) if state.missing_count_sources else np.zeros(len(x))
        for output, values in [('missing_count', count)] + ([('missing_fraction', count / max(1, len(state.missing_count_sources)))] if config['missing_fraction'] else []):
            state.means[output], state.stds[output] = float(values.mean()), max(float(values.std()), 1e-12)
            _add_output(state, output, ';'.join(state.missing_count_sources), 'raw_missingness_scaled', group='missingness')
    if config['engineer']:
        for sources in config['dummy_interactions'].values():
            for output in sources:
                if output in engineering_values:
                    continue
                if output not in state.output_features:
                    raise ValueError('Explicit dummy interaction source is unavailable: ' + output)
                found = False
                for source in state.numeric_features + state.binary_features:
                    if source_outputs.get(source) == output:
                        engineering_values[output] = engineering_values[source]
                        found = True
                        break
                if not found:
                    for source in state.categorical_features:
                        spec = state.specs[source]
                        if output in spec['outputs']:
                            values = _decode(x[:, positions[source]], state.special_code_map.get(source, {}))
                            tokens = category_tokens(values)
                            mapping = state.category_maps[source]
                            level = spec['levels'][spec['outputs'].index(output)]
                            engineering_values[output] = np.array([mapping[token] == level for token in tokens], dtype=float)
                            found = True
                            break
                if not found:
                    raise ValueError('Dummy interactions require retained numeric/binary/one-hot sources')
                source_outputs[output] = output
        state.engineered_feature_config = fit_engineering(engineering_values, source_outputs, config)
        for recipe in state.engineered_feature_config['recipes']:
            _add_output(state, recipe['name'], ';'.join(recipe['sources']), recipe['kind'] + '_scaled', group='engineered')
        for record in state.engineered_feature_config['skipped']:
            _manifest(state, ';'.join(record['sources']), 'engineering_skipped', output=record['name'],
                      reason=record['reason'], dropped=True, group='engineered')
    return state


def _target_01(y, n):
    y = np.asarray(y, dtype=float)
    if y.shape != (n,):
        raise ValueError('y must be 1-D with one value per training row')
    if np.isin(y, [0, 1]).all():
        return y
    if np.isin(y, [-1, 1]).all():
        return (y + 1) / 2
    raise ValueError('Target labels must use {0,1} or {-1,+1}')


def _target_options(config):
    return {'n_folds': config['target_folds'], 'seed': config['target_seed'], 'alpha': config['target_alpha'],
            'rare_min_count': config['rare_min_count'], 'rare_min_fraction': config['rare_min_fraction']}


def _transform(x, feature_names, state, *, target_values=None):
    x, names = _validate(x, feature_names)
    if set(names) != set(state.input_features):
        raise ValueError('Input feature names differ from the fitted schema')
    if state.target_features and target_values is None and len(x) == state.training_rows:
        if _signature(x, names) == state.training_signature:
            raise ValueError('Use fit_transform for OOF target-encoded training rows; transform is for unseen rows')
    estimated = len(x) * len(state.output_features) * np.dtype(state.dtype).itemsize
    if estimated > state.config['memory_warning_gib'] * 1024**3:
        warnings.warn(f'Dense processed allocation: {estimated / 1024**3:.2f} GiB', RuntimeWarning, stacklevel=2)
    result = np.zeros((len(x), len(state.output_features)), dtype=state.dtype)
    positions = {name: index for index, name in enumerate(names)}
    outputs = {name: index for index, name in enumerate(state.output_features)}
    unknown = {}
    target_columns = []
    for name in state.input_features:
        spec = state.specs[name]
        if spec['kind'] == 'dropped' and not spec.get('indicators'):
            continue
        values = _decode(x[:, positions[name]], state.special_code_map.get(name, {}))
        indicator_values = values
        if spec['kind'] == 'numeric':
            numeric, _ = _prepare_numeric(values, name, state.config, spec['numeric'])
            imputed = np.where(np.isnan(numeric), state.medians[name], numeric)
            result[:, outputs[spec['output']]] = 0 if spec['zero_variance'] else (imputed - state.means[name]) / state.stds[name]
        elif spec['kind'] == 'binary':
            tokens = category_tokens(values)
            mapped = np.array([spec['mapping'].get(token, np.nan) for token in tokens])
            unknown[name] = int(sum(token not in spec['mapping'] and token != MISSING for token in tokens))
            result[:, outputs[spec['output']]] = np.nan_to_num(mapped, nan=0.0)
            result[:, outputs[spec['other_output']]] = np.isfinite(values) & np.isnan(mapped)
        elif spec['kind'] == 'categorical':
            tokens = category_tokens(values)
            mapping = state.category_maps[name]
            mapped = np.array([mapping.get(token, MISSING if token == MISSING else OTHER) for token in tokens])
            unknown[name] = int(sum(token not in mapping and token != MISSING for token in tokens))
            for level, output in zip(spec['levels'], spec['outputs']):
                result[:, outputs[output]] = mapped == level
        elif spec['kind'] == 'target_encoded':
            target_columns.append((name, values))
        for kind in spec.get('indicators', []):
            result[:, outputs[name + '__' + kind]] = np.isfinite(indicator_values) if kind == 'present' else np.isnan(indicator_values)
    if state.target_features:
        columns = dict(target_columns)
        matrix = np.column_stack([columns[name] for name in state.target_features])
        encoded = transform_target_encoder(matrix, state.target_encoder, state.target_features) if target_values is None else target_values
        for index, name in enumerate(state.target_features):
            result[:, outputs[name + '__target']] = (encoded[:, index] - state.means[name]) / state.stds[name]
            known = state.category_maps[name]
            unknown[name] = int(sum(token not in known and token != MISSING for token in category_tokens(columns[name])))
    if 'missing_count' in outputs:
        count = sum(np.isnan(x[:, positions[name]]) for name in state.missing_count_sources)
        count = np.asarray(count, dtype=float) if state.missing_count_sources else np.zeros(len(x))
        result[:, outputs['missing_count']] = (count - state.means['missing_count']) / state.stds['missing_count']
        if 'missing_fraction' in outputs:
            fraction = count / max(1, len(state.missing_count_sources))
            result[:, outputs['missing_fraction']] = (fraction - state.means['missing_fraction']) / state.stds['missing_fraction']
    for recipe in state.engineered_feature_config.get('recipes', []):
        result[:, outputs[recipe['name']]] = apply_recipe(result, outputs, recipe)
    if not np.isfinite(result).all():
        raise ValueError('Processed matrix is nonfinite; check configured transforms and output dtype')
    return result, {'unknown_category_counts': unknown, 'unknown_category_total': sum(unknown.values()),
                    'rows': len(x), 'columns': result.shape[1], 'bytes': result.nbytes}


def transform(x, feature_names, state):
    """Transform unseen data, preserving state and output order; no targets needed."""
    return _transform(x, feature_names, state)[0]


def transform_with_diagnostics(x, feature_names, state):
    return _transform(x, feature_names, state)


def fit_transform(x, feature_names, *, y=None, config=None):
    """Return (processed training matrix, state); target columns are always OOF."""
    state = fit_preprocessor(x, feature_names, y=y, config=config)
    target_values = None
    if state.target_features:
        x, names = _validate(x, feature_names)
        columns = np.column_stack([_decode(x[:, names.index(name)], state.special_code_map.get(name, {})) for name in state.target_features])
        target_values, _ = cross_fit_target_encode(columns, _target_01(y, len(x)), state.target_features,
                                                   **_target_options(state.config))
    return _transform(x, feature_names, state, target_values=target_values)[0], state


def save_state(state, path):
    Path(path).write_text(json.dumps(asdict(state), indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load_state(path):
    values = json.loads(Path(path).read_text(encoding='utf-8'))
    if values.get('version') != 1:
        raise ValueError('Unsupported preprocessing state version')
    return PreprocessingState(**values)


def write_manifest(state, path):
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(state.manifest[0]))
        writer.writeheader()
        writer.writerows(state.manifest)
