# Exploratory Analysis

Standalone, reproducible EDA for the Project 1 training data. It characterizes
the full dataset, proposes preprocessing and feature-engineering experiments,
and ranks exactly three modeling strategies. It does not modify data, preprocess
a final matrix, train classifiers, or change the mandatory implementations.

## Run

Requires Python 3.10+, NumPy, and matplotlib (used only for visualization).

```powershell
python -m pip install -r "Exploratory Analysis/requirements.txt"
python -B "Exploratory Analysis/eda.py"
```

Or, from this folder:

```powershell
python -B eda.py
```

Defaults are resolved relative to the script, so the current working directory
does not affect input/output paths. Inputs default to `../dataset/x_train.csv`
and `../dataset/y_train.csv`; outputs default to `eda_outputs/` in this folder.

```powershell
python -B eda.py --data-dir "C:/path/to/data" --output-dir "C:/path/to/results" --sample-size 50000
```

The loader expects numeric, unquoted, single-line CSV records with an `Id`
column and blank/NaN/NA missing cells; IDs must be integer-valued and smaller
than 2^53 in magnitude. Headers can be reordered. Target columns must be `Id`
and `_MICHD`. Missing/duplicate IDs, unmatched ID sets, different ID ordering,
unexpected target values, malformed rows, and infinite numeric values stop the
analysis rather than silently producing associations on mismatched data.

## Outputs

Start with **[eda_outputs/eda_report.md](eda_outputs/eda_report.md)**.

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

## Checks

```powershell
python -B -m unittest discover -s "Exploratory Analysis" -p "test_*.py" -v
```

Checks cover CSV parsing, reordered columns, ID alignment failures, missingness
and category statistics, exact duplicate detection, and pairwise-complete
correlations against direct NumPy calculations. The full script should also be
run on the real training data to regenerate all report artifacts.
