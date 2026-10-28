"""Explicit policy, semantic overrides and configurable representation groups."""

import copy
import csv

import numpy as np


CODEBOOK_URL = 'https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf'
IDENTIFIERS = ['Id', 'SEQNO', '_PSU']
RAW_CONSTANTS = ['CTELENUM', 'COLGHOUS', 'STATERES', 'CTELNUM1', 'CELLFON2', 'CCLGHOUS']
EXACT_DUPLICATES = {'_PSU': 'SEQNO', 'CELLFON2': 'CTELNUM1', '_CPRACE': '_CRACE1'}
LEAKAGE_REVIEW_FEATURES = [
    'HAREHAB1', 'STREHAB1', 'CVDASPRN', 'RDUCHART', 'RDUCSTRK', 'CVDSTRK3',
    '_MICHD', 'CVDINFR4', 'CVDCRHD4',
    'DISPCODE', 'FMONTH', 'IDATE', 'IMONTH', 'IDAY', 'IYEAR', 'FLSHTMY2', 'HIVTSTD3',
    'QSTVER', 'QSTLANG', 'MSCODE', '_STSTR', '_STRWT', '_RAWRAKE', '_WT2RAKE',
    '_CLLCPWT', '_DUALCOR', '_LLCPWT',
]
# Individually verified in CDC 2015 codebook pp.13,16,105; no blanket code rule.
# Other fields retain values such as 7/9/88/99 unless explicitly configured.
VERIFIED_SPECIAL_CODES = {
    'GENHLTH': {7: 'missing', 9: 'missing'},
    'BPHIGH4': {7: 'missing', 9: 'missing'},
    'BPMEDS': {7: 'missing', 9: 'missing'},
    '_RFHYPE5': {9: 'missing'},
}
VERIFIED_BINARY_MAPS = {'BPMEDS': {1: 1, 2: 0}, '_RFHYPE5': {1: 0, 2: 1}}

REDUNDANCY_GROUPS = {
    'age': ['_AGE80', '_AGEG5YR', '_AGE65YR', '_AGE_G'],
    'height': ['HEIGHT3', 'HTIN4', 'HTM4'],
    'weight_bmi': ['WEIGHT2', 'WTKG3', '_BMI5', '_BMI5CAT', '_RFBMI5'],
    'general_health': ['GENHLTH', '_RFHLTH'],
    'blood_pressure': ['BPHIGH4', '_RFHYPE5', 'BPMEDS'],
    'physical_activity': ['EXERANY2', 'EXRACT11', 'EXEROFT1', 'EXERHMM1', 'EXRACT21',
                          'EXEROFT2', 'EXERHMM2', 'STRENGTH', 'PADUR1_', 'PADUR2_',
                          'PA1MIN_', 'PA1VIGM_', 'STRFREQ_', '_TOTINDA', '_PAINDX1', '_PASTRNG'],
    'smoking': ['SMOKE100', 'SMOKDAY2', 'STOPSMK2', 'LASTSMK2', '_SMOKER3', '_RFSMOK3'],
    'race': ['_CRACE1', '_CPRACE', '_PRACE1', '_MRACE1', '_RACE', '_RACEG21', '_RACEGR3', '_RACE_G1'],
    'oxygen_capacity': ['MAXVO2_', 'FC60_'],
    'alcohol_quantity': ['AVEDRNK2', '_DRNKWEK'],
}
GROUP_PROFILES = {
    'age': {'age80': ['_AGE80'], 'age_group_onehot': ['_AGEG5YR']},
    'height': {'meters': ['HTM4'], 'inches': ['HTIN4']},
    'weight_bmi': {'weight_bmi': ['WTKG3', '_BMI5'], 'weight_only': ['WTKG3'], 'bmi_only': ['_BMI5']},
    'general_health': {'general_health_raw': ['GENHLTH'], 'general_health_derived': ['_RFHLTH'],
                       'general_health_both': ['GENHLTH', '_RFHLTH']},
    'blood_pressure': {'raw': ['BPHIGH4', 'BPMEDS'], 'derived': ['_RFHYPE5', 'BPMEDS'],
                       'both': ['BPHIGH4', '_RFHYPE5', 'BPMEDS']},
    'physical_activity': {
        'activity_raw': ['EXERANY2', 'EXRACT11', 'EXEROFT1', 'EXERHMM1', 'EXRACT21', 'EXEROFT2', 'EXERHMM2', 'STRENGTH'],
        'activity_derived': ['_TOTINDA', 'PA1MIN_', 'PA1VIGM_', '_PAINDX1', '_PASTRNG'],
        'activity_compact': ['_TOTINDA', '_PAINDX1', '_PASTRNG'],
    },
    'smoking': {'raw': ['SMOKE100', 'SMOKDAY2', 'STOPSMK2', 'LASTSMK2'], 'derived': ['_SMOKER3']},
    'race': {name: [name] for name in REDUNDANCY_GROUPS['race'] if name != '_CPRACE'},
}

DEFAULT_CONFIG = {
    'profile': 'curated_v1', 'safe_removals': True, 'missingness': True,
    'encode_categorical': True, 'canonical': True, 'engineer': False, 'target_encoding': False,
    'max_onehot_levels': 5, 'onehot_limits': {}, 'rare_min_count': 100, 'rare_min_fraction': 0.001,
    'sparse_threshold': 0.95, 'missing_indicator_threshold': 0.01,
    'include_quarantined': False, 'quarantine_features': LEAKAGE_REVIEW_FEATURES,
    'output_dtype': 'float32', 'memory_warning_gib': 2.0, 'preserve_presence': True,
    'missing_fraction': False, 'explicit_missing_indicators': ['BPMEDS'],
    'special_code_map': VERIFIED_SPECIAL_CODES, 'binary_maps': VERIFIED_BINARY_MAPS,
    # CDC 2015 codebook pp.113-115. The project CSV already contains decimal
    # HTM4/WTKG3/_BMI5 values; do not reapply the source file's implied decimals.
    'semantic_types': {'_AGE80': 'numeric', 'HTM4': 'numeric', 'HTIN4': 'numeric',
                       'WTKG3': 'numeric', '_BMI5': 'numeric'},
    'type_hints': {}, 'allow_inferred_numeric': False,
    'group_choices': {'age': 'age80', 'height': 'meters', 'weight_bmi': 'weight_bmi',
                      'general_health': 'general_health_raw', 'blood_pressure': 'raw',
                      'physical_activity': 'activity_compact', 'smoking': 'derived', 'race': '_RACEG21'},
    'redundancy_groups': REDUNDANCY_GROUPS, 'group_profiles': GROUP_PROFILES,
    'exact_duplicates': EXACT_DUPLICATES, 'log1p_features': [], 'clip_features': {},
    'squares': {'age_squared': '_AGE80', 'BMI_squared': '_BMI5'},
    'interactions': {}, 'dummy_interactions': {}, 'major_condition_features': [],
    'target_features': None, 'max_target_levels': 50, 'target_alpha': 20.0,
    'target_folds': 5, 'target_seed': 42,
}


def get_config(profile='curated_v1', **overrides):
    """Return a validated copy of the default config for a named profile, with overrides."""
    if profile not in {'minimal', 'curated_v1', 'curated_v1_engineered', 'compact_target_encoded'}:
        raise ValueError('Unknown preprocessing profile: ' + str(profile))
    config = copy.deepcopy(DEFAULT_CONFIG)
    config['profile'] = profile
    if profile == 'minimal':
        config.update(safe_removals=False, missingness=False, encode_categorical=False, canonical=False)
    elif profile == 'curated_v1_engineered':
        config['engineer'] = True
    elif profile == 'compact_target_encoded':
        config['target_encoding'] = True
    unknown = set(overrides) - set(config)
    if unknown:
        raise ValueError('Unknown preprocessing settings: ' + ', '.join(sorted(unknown)))
    config.update(copy.deepcopy(overrides))
    config['output_dtype'] = np.dtype(config['output_dtype']).name
    if config['output_dtype'] not in ('float32', 'float64'):
        raise ValueError('output_dtype must be float32 or float64')
    for name in ['rare_min_fraction', 'sparse_threshold', 'missing_indicator_threshold']:
        if not 0 <= config[name] <= 1:
            raise ValueError(name + ' must be in [0, 1]')
    for name in ['rare_min_count', 'max_onehot_levels', 'max_target_levels', 'target_folds']:
        if not isinstance(config[name], int) or config[name] < 1:
            raise ValueError(name + ' must be a positive integer')
    if config['memory_warning_gib'] <= 0:
        raise ValueError('memory_warning_gib must be positive')
    if any(kind not in {'numeric', 'binary', 'categorical'} for kind in config['semantic_types'].values()):
        raise ValueError('semantic_types values must be numeric, binary or categorical')
    return config


def load_eda_hypotheses(path):
    """Read candidate types only; never import EDA medians, counts or target scores."""
    with open(path, newline='', encoding='utf-8') as stream:
        return {row['feature_name']: row['candidate_feature_type'] for row in csv.DictReader(stream)}


def load_semantic_review(path):
    """Read the feature names listed in a semantic-review CSV."""
    with open(path, newline='', encoding='utf-8') as stream:
        return [row['feature_name'] for row in csv.DictReader(stream)]
