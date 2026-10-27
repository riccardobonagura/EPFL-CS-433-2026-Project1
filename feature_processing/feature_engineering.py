"""Small explicit recipes operating on validated numeric/binary representations."""

import numpy as np


def fit_engineering(values, source_outputs, config):
    """Values map source names to train-fitted numeric/binary output vectors."""
    recipes, skipped = [], []
    candidates = [(name, [source, source], 'square') for name, source in config['squares'].items()]
    candidates += [(name, list(sources), 'interaction') for name, sources in config['interactions'].items()]
    candidates += [(name, list(sources), 'explicit_dummy_interaction') for name, sources in config['dummy_interactions'].items()]
    for name, sources, kind in candidates:
        if len(sources) != 2 or any(source not in values for source in sources):
            skipped.append({'name': name, 'sources': sources, 'reason': 'unvalidated_or_unavailable_numeric_source'})
            continue
        product = values[sources[0]] * values[sources[1]]
        recipes.append({'name': name, 'sources': [source_outputs[source] for source in sources], 'kind': kind,
                        'mean': float(product.mean()), 'std': max(float(product.std()), 1e-12)})
    conditions = config['major_condition_features']
    if conditions:
        if any(source not in config['binary_maps'] or source not in values for source in conditions):
            raise ValueError('major_condition_count requires explicitly verified, retained binary sources')
        count = sum(values[source] for source in conditions)
        recipes.append({'name': 'major_condition_count', 'sources': [source_outputs[s] for s in conditions],
                        'kind': 'sum', 'mean': float(count.mean()), 'std': max(float(count.std()), 1e-12)})
    return {'recipes': recipes, 'skipped': skipped}


def apply_recipe(matrix, positions, recipe):
    if recipe['kind'] == 'sum':
        values = sum(matrix[:, positions[source]].astype(float) for source in recipe['sources'])
    else:
        values = matrix[:, positions[recipe['sources'][0]]].astype(float) * matrix[:, positions[recipe['sources'][1]]]
    return (values - recipe['mean']) / recipe['std']
