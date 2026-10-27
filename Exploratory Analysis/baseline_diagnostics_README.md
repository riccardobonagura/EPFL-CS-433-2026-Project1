# Mandatory baseline diagnostics

Run the separate diagnostic from the repository root:

```powershell
python -B "Exploratory Analysis/baseline_diagnostics.py"
```

Or run `python -B baseline_diagnostics.py` from this folder. Dependencies are
the existing NumPy and matplotlib requirements; no additional packages are used.
The script neither imports nor runs the main EDA, and does not modify its outputs
or the six mandatory functions.

The default sample contains 100,000 rows, selected proportionally by class with
seed 42. All six functions share one stratified 80/20 split. Training-only median
imputation and standardization retain every predictor except Id; an intercept is
added. Zero-variance training columns remain as zero columns, and all-missing
training columns use zero imputation. Numeric survey response codes stay observed.
This deliberately crude numeric representation is not the final feature pipeline.

```powershell
python -B "Exploratory Analysis/baseline_diagnostics.py" --max-rows 100000
```

Optional `--data-dir` and `--output-dir` paths are supported. Defaults are relative
to the script rather than the working directory. Data and the protected EDA and
mandatory-function directories cannot be used as output directories.

Results are saved only in `baseline_diagnostics/`:

- **[baseline_report.md](baseline_diagnostics/baseline_report.md)**: interpretation
  of validation, convergence, conditioning, class imbalance, and runtime evidence.
- `model_metrics.csv`: every attempted mandatory configuration and majority reference.
- `convergence_summary.csv`: independently restarted iterative runs.
- `predictions_summary.csv`: validation score/probability ranges and positive fractions.
- `preprocessing_summary.txt` and `preprocessing_parameters.csv`: exact train-only procedure/statistics.
- `run_metadata.json`: configuration, seeds, split/input fingerprints, and software versions.
- `figures/`: five diagnostic comparisons.

There are 23 mandatory configurations: GD at 10/50/100 updates; size-one SGD at
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

Run focused checks without running the main EDA:

```powershell
python -B -m unittest discover -s "Exploratory Analysis" -p "test_baseline_diagnostics.py" -v
```
