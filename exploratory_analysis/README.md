# exploratory_analysis

Two standalone, reproducible analyses of the Project 1 training data:

1. **EDA** (`eda.py`): characterizes the full dataset, proposes preprocessing and
   feature-engineering experiments, and ranks three modeling strategies.
2. **Baseline diagnostics** (`baseline_diagnostics.py`): runs the six required
   methods on a crude numeric representation to check them on real data.

Neither modifies data, builds the final feature matrix, or changes
`implementations.py`. The two scripts are independent of each other.

## Setup

Requires Python 3.10+, NumPy, and matplotlib (used only for visualization).

```powershell
python -m pip install -r exploratory_analysis/requirements.txt
```

## EDA

```powershell
python -B exploratory_analysis/eda.py
```

Or, from this folder, `python -B eda.py`. Defaults are resolved relative to the
script, so the current working directory does not affect input/output paths.
Inputs default to `../dataset/x_train.csv` and `../dataset/y_train.csv`; outputs
default to `eda_outputs/` in this folder.

```powershell
python -B eda.py --data-dir "C:/path/to/data" --output-dir "C:/path/to/results" --sample-size 50000
```

The loader expects numeric, unquoted, single-line CSV records with an `Id`
column and blank/NaN/NA missing cells; IDs must be integer-valued and smaller
than 2^53 in magnitude. Headers can be reordered. Target columns must be `Id`
and `_MICHD`. Missing/duplicate IDs, unmatched ID sets, different ID ordering,
unexpected target values, malformed rows, and infinite numeric values stop the
analysis rather than silently producing associations on mismatched data.

### EDA outputs

Start with **[eda_outputs/eda_report.md](eda_outputs/eda_report.md)** (generated, git-ignored).

- `feature_summary.csv`: every predictor, full-data statistics, candidate type,
  removal-review reasons, and proposed preprocessing flags.
- `target_summary.txt`: counts, proportions, and majority-class accuracy.
- `missingness_target.csv`: observed/missing support, positive rates and contrasts.
- `category_target_summary.csv`: categorical frequency, class rate and support.
- `target_associations.csv`: per-class statistics and exploratory ranking components.
- `potential_special_codes.csv`: repeated-digit response values for codebook review.
- `semantic_review.csv`: provenance/leakage review candidates, without asserting leakage.
- `exact_duplicate_columns.csv`, `categorical_agreement.csv`, `correlated_pairs.csv`,
  `missingness_patterns.csv`: redundancy diagnostics with scope/support.
- `analysis_metadata.json`: thresholds, sample sizes, seed, versions and integrity checks.
- `figures/`: eight readable, reproducible figures linked from the report.

All feature statistics, quantiles and target associations use the full training
set. Exact duplicates include matching NaN positions. Only correlations,
categorical near-duplicate comparisons and distribution plots use deterministic
samples. Pairwise-complete correlations do not impute missing data.

The float64 input matrix alone occupies roughly 0.8 GiB on the supplied data;
allow a few GiB of RAM for chunks, samples and temporary arrays. CSV loading
and full-data profiling may take several minutes. Chunked loading bounds parsing
memory, and column-contiguous storage makes per-feature operations efficient.
Existing lab loaders were inspected; their combined-file/string-label or demo
dataset layouts do not match these separate training CSVs.

Interpret numeric categories and survey codes with the BRFSS codebook before
using them in models. Candidate types are documented heuristics. Association,
causality, predictive performance and leakage are different claims; this EDA
does not establish the latter three. Full-data target associations are discovery
diagnostics; repeat selection within training folds for honest model comparisons.

## Baseline diagnostics of the six methods

```powershell
python -B exploratory_analysis/baseline_diagnostics.py
python -B exploratory_analysis/baseline_diagnostics.py --max-rows 100000
```

Or run `python -B baseline_diagnostics.py` from this folder. Optional `--data-dir`
and `--output-dir` paths are supported. Defaults are relative to the script
rather than the working directory. Data, `implementations.py` and the EDA output
folder cannot be used as output directories.

The default sample contains 100,000 rows, selected proportionally by class with
seed 42. All six functions share one stratified 80/20 split. Training-only median
imputation and standardization retain every predictor except Id; an intercept is
added. Zero-variance training columns remain as zero columns, and all-missing
training columns use zero imputation. Numeric survey response codes stay observed.
This deliberately crude numeric representation is not the final feature pipeline.

There are 23 configurations: GD at 10/50/100 updates; size-one SGD at
1,000/10,000/100,000 updates; one direct least-squares call; ridge at lambda
0/0.0001/0.01/1; logistic at 10/50/100 updates; and regularized logistic at those
three budgets for lambda 0.0001/0.01/1. All iterative fits start from zero.
Training curvature bounds choose fixed step sizes; there is no validation step-size
search or decision-threshold tuning. SGD restarts with the same random sequence
for each budget. Metrics are deterministic; wall-clock timings vary.

Retained constant and redundant columns can make the course normal-equation
solvers singular. Such failures remain visible in the report/CSVs with blank
metrics. The diagnostic calls the required functions without replacing their
solvers, selecting columns, or treating rank deficiency as an implementation bug.
Successful configurations are audited for finite results and correct final losses.

### Baseline outputs

Results are saved only in `baseline_diagnostics/` (generated, git-ignored):

- **[baseline_report.md](baseline_diagnostics/baseline_report.md)**: interpretation
  of validation, convergence, conditioning, class imbalance, and runtime evidence.
- `model_metrics.csv`: every attempted configuration and the majority reference.
- `convergence_summary.csv`: independently restarted iterative runs.
- `predictions_summary.csv`: validation score/probability ranges and positive fractions.
- `preprocessing_summary.txt` and `preprocessing_parameters.csv`: exact train-only procedure/statistics.
- `run_metadata.json`: configuration, seeds, split/input fingerprints, and software versions.
- `figures/`: five diagnostic comparisons.

## Tests

```powershell
python -B -m unittest discover -s exploratory_analysis -p "test_*.py" -v
```

`test_eda.py` covers CSV parsing, reordered columns, ID alignment failures,
missingness and category statistics, exact duplicate detection, and
pairwise-complete correlations against direct NumPy calculations.
`test_baseline_diagnostics.py` covers sampling, training-only preprocessing,
metrics and the calls to the six methods. The full scripts should also be run on
the real training data to regenerate all report artifacts.
