# Three Project 1 experiments

These experiments compare tuned logistic representations, selective OOF target
encoding, and early-stopped MSE-SGD. A0/minimal is the benchmark to beat, not a
representation to discard because more elaborate processing exists.

Only NumPy and Python standard-library modules are used. Figures are exported
as both PNG and SVG without additional plotting dependencies. Existing feature
processing and augmented fitters, metrics and threshold helpers are reused; no
ML algorithm is reimplemented and neither ML implementation file is modified.

## Run

From the repository root:

```powershell
python -B -m ml_attempts.run_all
```

Or run an attempt independently:

```powershell
python -B ml_attempts/01_logistic_representation/run.py
python -B ml_attempts/02_logistic_target_encoded/run.py
python -B ml_attempts/03_mse_sgd/run.py
```

Attempts 2/3 reuse a matching saved Attempt 1 selection. If it is unavailable,
they run the prerequisite Attempt 1 exploration automatically. No permission
prompt or external submission is involved. Training can take substantial time;
each configuration is printed and its CSV row/weights/history are saved promptly.

Phases support concrete inspection and resuming completed work:

```powershell
python -B -m ml_attempts.run_all --phase explore
python -B -m ml_attempts.run_all --phase finish
```

`explore` performs the staged searches only. `finish` loads matching selections,
confirms finalists not already confirmed, then refits/exports. `confirm` reruns
confirmation only; `final` requires existing confirmation and performs final
refits only. The default `all` performs the complete workflow. All requested
finalists finish confirmation before any requested test inference begins.

`--max-rows N` permits an explicitly labeled smaller shared diagnostic sample;
default zero uses all rows. `--max-logistic-updates N` and `--max-mse-updates N`
permit bounded smoke runs. Different sample scopes are recorded in split/run
metadata; results from a mismatched split are not reused. Config files hold the
default full-search settings. Raw inputs are read only from this repository's
`dataset/`; experiment outputs do not overwrite earlier EDA/preprocessing outputs.
Update overrides affect exploration; finishing uses the saved selected settings.
An explicit logistic-budget override reruns the prerequisite benchmark even for
an independently launched Attempt 2/3, so a saved long-budget run cannot silently
override a requested short diagnostic.

## Shared split and targets

The existing seeded stratified sampling/splitting functions reproduce the full
preprocessing split exactly: sampling seed 42, split seed 43, approximately 80/20.
On the supplied files this is 262,508 training and 65,627 validation rows. Stored
indices in `shared_split/train_indices.npy` and `validation_indices.npy` address
original zero-based CSV rows, not positions in a temporary permutation. Input
ID presence, uniqueness and exact X/y alignment are verified, and file/split
hashes prevent accidental reuse on changed data.

Logistic receives explicit `(y_pm1 + 1)/2` labels. MSE receives {-1,+1} directly.
Both use augmented binary metrics with positive label 1. Accuracy, balanced
accuracy, F1, precision, recall, specificity and all confusion counts are saved,
alongside actual/predicted positive fractions. Selection ranks balanced accuracy,
then F1, recall and precision; exact metric ties prefer fewer performed updates.
No run is selected on ordinary accuracy alone.

Every run reports the standard threshold (0.5 probability or 0 linear score) and
its independently validation-selected threshold. The earlier 0.1135 threshold is
historical context only and is never assigned to new predictions. Threshold
selection does not modify weights. Raw MSE and logistic loss are never compared
as equivalent optimization objectives.

## Search plans

1. **Logistic representation:** R0=A0, R1=A2, R2=curated V1 are obtained directly
   from existing preprocessing ablation profiles. For each, four lambdas are
   tested at gamma 0.03, then gamma 0.01/0.1 at the best lambda. This is 18
   configurations, not a full Cartesian sweep. Mini-batch size 1024, at most
   5,000 updates, evaluation every 100, patience 5 and tolerance 1e-5 give bounded
   fixed-step experiments. Actual updates, best validation iteration/loss and
   stopping reason are always recorded. Engineering is disabled by default;
   an explicit optional one-block comparison requires a material gain to win.
2. **Target-encoding recovery:** TE1 uses the unengineered compact curated
   profile. TE2 starts from the actual Attempt 1 winner and adds at most eight
   predeclared nominal fields filtered by training-only unused-medium status,
   support and quarantine. Only that supervised block uses the existing OOF
   encoder. Full-data vocabulary/rare groups/prior are never used for a training
   row's TE value. Alpha starts at 20; only competitive variants get alpha 5/100
   checks. Each variant tests lambda*/10, lambda*, 10*lambda* at the winning
   Attempt 1 gamma/settings. If the base is minimal, raw numeric-code versions
   can coexist with the added categorical target rates; this is extra nonlinear
   information rather than falsely described restoration of absent raw columns.
3. **MSE-SGD:** the actual winning Attempt 1 recipe is used first and throughout.
   Three gammas on batch 256 precede comparison of batch sizes 1/256/1024 and one
   inverse schedule. A failed size-one run may receive one training-norm-bounded
   step-size diagnostic. Independent budget prefixes replay the same RNG/schedule
   from zero, always with validation-MSE early stopping. Classification/threshold
   metrics select among the returned prefix checkpoints; there is no local SGD
   reimplementation, fake dense classification history, or RNG-reset warm start.
   Requested budgets differ from returned best-MSE checkpoint iterations; both
   are saved. A low-cost size-one run may represent much less than one data pass.

No class weighting, quarantine reintroduction, exhaustive alpha search, blind
engineering block, final-model hyperparameter sweep or leaderboard tuning occurs.
The profiles remain conservative hypotheses: success is not assumed, and tiny
0.0001-scale validation differences are not described as decisive improvements.

## Confirmation and final refits

Exactly one finalist per attempt receives the same three stratified outer folds
(seed 2026). Each outer-training fold has a shared 20% inner stopping/calibration
partition. Preprocessing—including OOF target encoding—is fitted only on the
remaining inner-training rows. Inner labels select stopping and threshold; outer
labels only score the result. No outer labels enter category selection, encoder
priors or thresholds. This deliberately fits about 53% of the source per fold,
without an outer-training refit after calibration. It is conservative compared
with the 80%-training exploratory fits. Recipes were selected earlier on
overlapping source rows, so this is conditional finalist confirmation, not a
fully nested audit of the entire search. Fold means/SDs and IDs are retained.

After confirmation, each experiment freezes its recipe, hyperparameters, median
inner-CV threshold and median selected data-pass budget. It fits preprocessing
and model weights on **all** labeled training rows. With no validation rows left,
this refit disables early stopping and uses the previously selected budget;
inverse decay is rescaled consistently with data passes. It does not tune anything
against test predictions. Only then is `x_test.csv` read for inference.

Each experiment exports its own `final_model_weights.npz`, complete reusable
`preprocessing_state/`, `test_predictions.csv` and `submission.csv`. Submissions
contain `Id,_MICHD`, preserve original test ID order, and contain only -1/+1
labels. The repository has no matching submission helper, so a validated stdlib
CSV writer supplies this exact format. Files stay local; nothing is uploaded.

The preprocessing-state directory stores the core preprocessing JSON/manifest
and a representation JSON. Selective recovery additionally stores its final
full-training encoder, selected source names and ordered output dimensions.

## Reports and files

Each `outputs/` includes `all_runs.csv`, `validation_metrics.csv`,
`threshold_results.csv`, `run_config.txt`, `experiment_report.md`, selected
configuration metadata, run-level weights/scores/histories, stage-specific tables,
figures, confirmation fold artifacts and final inference artifacts. The model
matrices are float64 (the augmented fitter's working precision), while the
underlying preprocessing state retains its float32 default. Matrix-memory tables
report the actual train-plus-validation model matrices, including intercepts.
Raw inputs and temporary arrays need additional memory; full-data work needs a
few GiB. Only one representation is kept in working memory at a time.

`model_comparison.csv` and `summary.md` summarize the three selected configurations
and their outer-fold confirmation. Generated outputs, split indices and master
summaries are ignored by Git; code and documentation remain trackable.

```powershell
python -B -m unittest ml_attempts.test_experiments -v
```

Tests cover exact split reuse, OOF recovery independence from a row's own label,
quarantine, shared metrics, the confirmation-before-test guard, fold separation,
all-three-attempt execution, full-row refitting, transparent state, final threshold
freezing, original test ID order, labels, immutable input CSVs and PNG/SVG output.
