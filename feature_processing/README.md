# Feature processing

This package fits preprocessing **only on a training partition**, stores a
transparent state, and applies that state to validation/test matrices. The ML
implementations and raw CSVs are never modified. Dependencies are NumPy and the
standard library; evaluation figures additionally use matplotlib.

## Run

From the repository root:

```powershell
python -B -m feature_processing.run_preprocessing
python -B -m feature_processing.evaluate_processing
python -B -m unittest feature_processing.test_preprocessing -v
```

The processing runner defaults to all 328,135 training-file rows with a seeded
stratified 80/20 split. Evaluation defaults to a stratified 100,000-row sample
and the same sampling/splitting algorithm/seeds as the earlier baseline
diagnostics. All ablations share exactly one sample/split. Use `--max-rows N`
to bound either run; zero means all rows. Absolute defaults are derived from the
package location, so direct `python feature_processing/run_preprocessing.py`
also works from any working directory.

```powershell
python -B -m feature_processing.run_preprocessing --profile curated_v1_engineered --max-rows 100000
python -B -m feature_processing.run_preprocessing --save-arrays --test-csv dataset/x_test.csv
python -B -m feature_processing.evaluate_processing --max-rows 100000 --max-updates 1500
```

Test rows, if supplied, are transformed only after training-state fitting. They
never choose categories, parameters or representation policies. Their IDs remain
in original file order. Optional arrays are saved as `.npz`/`.npy` with no pickle.
Outputs default to `feature_processing/outputs/`, which Git ignores.

## In-memory API

```python
from feature_processing import (
    get_config, fit_transform, fit_preprocessor, transform,
    transform_with_diagnostics, save_state, load_state,
)

config = get_config("curated_v1")
x_train_processed, state = fit_transform(
    x_train, feature_names, y=y_train, config=config
)
x_val_processed, diagnostics = transform_with_diagnostics(
    x_val, feature_names, state
)
save_state(state, "preprocessing_state.json")
reloaded = load_state("preprocessing_state.json")
x_test_processed = transform(x_test, feature_names, reloaded)
```

`fit_preprocessor` returns state; `transform` returns a matrix; `fit_transform`
returns `(training_matrix, state)`. `transform_with_diagnostics` additionally
returns unknown-category counts and memory/shape information. Input names may be
reordered at transform time, but the fitted schema must match. All output names
and their order are fixed. NaNs are supported in raw numeric matrices, infinities
are rejected, and ordinary processed matrices are finite float32 by default.
Use `output_dtype="float64"` (or `np.float64`) explicitly when needed.

IDs are kept separately by the loaders and are never predictors, even if `Id`
is explicitly passed to the core API. `SEQNO` and `_PSU` are always excluded.
The package does **not** add an intercept. The evaluator adds one immediately
before calling the augmented logistic fitter with `penalize_intercept=False`.

## Profiles and conservative semantics

| Profile | Representation |
| --- | --- |
| `minimal` | Numeric median/scaling reference; no encoding, engineered terms or sparse/constant processing |
| `curated_v1` | Safe removals, verified decoding/binary maps, bounded one-hot, numeric treatment, missingness, quarantine and canonical groups |
| `curated_v1_engineered` | Curated V1 plus only the available configured validated recipes |
| `compact_target_encoded` | Curated V1 plus cross-fitted target encoding for selected medium-cardinality categories |

All profiles share identifier exclusion and the same default quarantine, including
administrative/date/weight metadata and the named cardiovascular review fields.
Thus `minimal` deliberately approximates the previous crude baseline rather than
reproducing its entire input set. `include_quarantined=True` is an explicit opt-in;
it is never enabled automatically. The full earlier EDA review list is broader:
use `--semantic-review "exploratory_analysis/eda_outputs/semantic_review.csv"` to
add it to quarantine. This can exclude age/BMI and many other derived fields;
review the manifest rather than assuming those names establish leakage.

Existing EDA candidate-type strings can be loaded via `--eda-summary`. Only the
type hypotheses are imported: all cardinalities, frequencies, vocabularies,
medians, means, clipping bounds and rare mappings are relearned from the current
training partition. No target association ranking is imported.

Curated numeric semantics default only to `_AGE80`, `HTM4`, `HTIN4`, `WTKG3`,
and `_BMI5`. Other numeric-looking inputs require explicit `semantic_types`
overrides or `allow_inferred_numeric=True`; unsupported medium categories and
unverified numeric fields are documented exclusions, not silently ordered codes.
This conservative default can discard useful information and is an experiment,
not a claim of optimal feature selection.

The [CDC 2015 BRFSS codebook](https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf)
(pp.13,16,105,113–115) supports the small default semantic/code rules. Only
GENHLTH, BPHIGH4 and BPMEDS codes 7/9 and _RFHYPE5 code 9 are decoded to missing.
BPMEDS states 1/2 map to 1/0; _RFHYPE5 states 1/2 map to 0/1. Other fields do
**not** inherit those mappings. The supplied CSV already has decimal derived
height/weight/BMI values; no implied-decimal rescaling is repeated. The codebook
does not by itself establish prediction-time availability or absence of leakage.

Custom verified rules have this form:

```python
config["special_code_map"] = {
    "verified_feature": {88: "missing", 99: "category", 77: 0.0}
}
config["semantic_types"] = {"verified_feature": "categorical"}
```

This example declares API structure only, not actual survey-code meanings.
Rules are applied simultaneously to raw values, before typing/imputation/scaling.
`"category"` preserves the code and forces categorical handling. Unlisted codes
remain observed. Binary features use verified maps, zero-fill missing/unknown
values, and retain distinct missing and unknown representations without scaling.

## Categories, missingness and redundancy

Rare support is `max(rare_min_count, ceil(rare_min_fraction * training_rows))`;
defaults are 100 and 0.001. Rare observed categories map to `__OTHER__` and
actual missing values to `__MISSING__`. The missing category is never merged
with rare/unknown values. Vocabularies, reference levels and rare groups are
stored in state, not reconstructed on validation data.

`max_onehot_levels=5` bounds effective **nonmissing** levels: retained ordinary
levels plus an observed rare group. Missing and unknown reserves are additional,
so a five-level feature can have up to six output dummies after omitting one
ordinary reference. The omission never merges missing with other. Reserved
columns exist even if absent during fit, keeping future transformations stable.
`onehot_limits={"name": K}` explicitly permits larger vocabularies; choosing
`age_group_onehot` also explicitly allows up to 20 age-group levels.

Numeric median imputation/scaling is training-only. Separate numeric missing
flags use a configurable 1% threshold or an explicit feature list. Categorical
missingness uses its missing dummy instead; separate flags require opt-in.
BPMEDS explicitly preserves applicability/missingness. Standalone flags with
identical training masks are deduplicated; ordinary per-category missing columns
are retained as part of their categorical schema.

Observed constants lose their raw value representation. Extremely sparse raw
values (>95% missing by default) are dropped. Nontrivial presence indicators can
survive these removals, and duplicate standalone indicators are not emitted.
Declared exact duplicates are checked for full training equality including NaNs;
mere high correlation does not trigger this rule. The selected representative
and `exact_duplicate` reason appear in the manifest.

`missing_count` counts original, actual raw NaNs before decoding or imputation.
Its eligible original sources exclude identifiers, configured duplicate copies
and quarantine, so quarantined presence cannot leak through this aggregate.
The count is standardized with training statistics. Optional `missing_fraction`
uses the same stored source set. This is a deliberate safety refinement of an
all-raw-column count, and the exact source list is saved in state.

Concept groups in `feature_config.py` cover age, height, weight/BMI, general
health, blood pressure, activity, smoking and race. They use explicit selectable
lists, never averages or correlation-only pruning. Example choices:

```python
config["group_choices"]["age"] = "age_group_onehot"  # or age80
config["group_choices"]["general_health"] = "general_health_derived"
config["group_choices"]["physical_activity"] = "activity_raw"
config["group_choices"]["weight_bmi"] = "weight_only"
```

These are representation experiments; profile names alone do not verify every
member's semantic type. Consult the manifest for unavailable/unvalidated sources.

## Controlled engineering and optional numeric transforms

The default engineered candidates are `age_squared` and `BMI_squared`, using
the standardized validated sources, then training-normalizing each product.
Interactions are explicitly listed, e.g.:

```python
config["interactions"] = {
    "age_x_general_health": ["_AGE80", "GENHLTH"],
    "age_x_hypertension": ["_AGE80", "_RFHYPE5"],
    "age_x_smoking": ["_AGE80", "SMOKE100"],
    "age_x_diabetes": ["_AGE80", "DIABETE3"],
}
```

Each source must actually have a validated numeric or verified binary retained
representation; unavailable/categorical sources are skipped and documented.
No ordinal scores or disease flags are invented to make an interaction work.
`dummy_interactions` allows explicitly named output-column pairs, such as
`["_AGE80__scaled", "GENHLTH__2"]`, without expanding all dummies automatically.
`major_condition_features` is empty by default; a condition count requires an
explicit list of retained verified binary maps. Missing binary values use the
documented zero baseline plus their applicability indicators.

`log1p_features` and `clip_features` are empty by default. Explicit clipping
percentiles are fitted on observed, decoded training values and reused later.
Log1p rejects negative decoded values; clipping is not used to hide them.
All transformed values are imputed and standardized after these optional steps.

## Target encoding safety

`target_encoding.py` exposes `fit_target_encoder`, `transform_target_encoder`
and `cross_fit_target_encode`. Smoothing is `(category_positive_sum + alpha*prior)
/(category_count + alpha)`. Labels are {0,1} internally; the central preprocessing
API accepts either {0,1} or {-1,+1}. Missing and other groups remain separate.

Training target encoding **must use `fit_transform`**. Five deterministic,
label-independent folds each learn their own prior, category vocabulary, rare
groups and rates using only the other folds. The saved state contains a separate
full-training encoder for **unseen** validation/test data. OOF probabilities are
left unscaled, avoiding a global normalization that could indirectly transmit a
row's target through other folds' encodings.

`transform` is for unseen data. It rejects an exact replay of the fitted training
matrix in target-encoded mode, including reordered columns, as a common misuse
guard. Never use it to encode arbitrary training subsets either: it intentionally
uses the full-training encoder. Use `fit_transform` for every training partition;
no target labels or OOF training arrays are serialized in the state.

## Outputs and evaluation

The runner writes `processing_report.md`, `feature_manifest.csv`,
`preprocessing_state.json` and `processing_run.json`. Optional arrays are not
needed for in-memory use. JSON reload reproduces validation output exactly.
Before allocating a dense output, rows × columns × dtype bytes are computed;
allocations above `memory_warning_gib=2.0` raise a visible warning. Raw input,
temporary matrices and the augmented model's float64 copies need additional RAM.

The evaluator writes `ablation_results.csv`, `ablation_report.md`, run metadata,
two figures, and individual states/manifests/convergence histories under
`outputs/ablations/A0` ... `A6`. Stages build cumulatively: minimal → safe removals
→ missingness/sparsity → semantic/binary/category encoding → canonical groups
→ controlled engineering → optional OOF target encoding. A3 includes a semantic
policy bundle, not a single isolated one-hot change; this is stated in the report.

The model settings, sample, split, seeds, mini-batch size, regularization and
stopping parameters remain fixed across stages. No class weighting or broad
hyperparameter search is performed. Both default-threshold and validation-selected
metrics are recorded. Validation selects stopping, thresholds and representation;
these are exploratory comparisons that need confirmation on fresh folds/holdout.
Neither runner fits a final production model or submits predictions.
