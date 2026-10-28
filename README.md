# EPFL CS-433 Project 1: predicting coronary heart disease (MICHD)

Binary classification of MICHD (heart attack, angina or coronary heart disease) from
the BRFSS 2015 survey: 328,135 labelled respondents, 321 raw fields, 8.8% positive.
Only Python, NumPy and the standard library are used; matplotlib is used only for EDA figures.

**Final model:** L2-regularized logistic regression fitted by Newton's method on 619
engineered features (Attempt 5, block F4), with an F1-maximizing decision threshold.

| Estimate | F1 | Accuracy |
| --- | --- | --- |
| 3-fold cross-validation (mean of outer folds) | 0.428 | 0.867 |
| AIcrowd leaderboard (public test set) | 0.442 | |

## Quick start

1. Put the course files `x_train.csv`, `y_train.csv` and `x_test.csv` in `dataset/`
   (keep the names; the folder is git-ignored).
2. Install NumPy (`python -m pip install numpy`; tested with Python 3.10+).
3. From the repository root:

```text
python run.py
```

This writes `submission.csv`, byte-identical to our best leaderboard submission, in
about 1-2 minutes. It trains the final model on all labelled rows using the frozen
choices in [final_model/frozen_configuration.json](final_model/frozen_configuration.json)
(feature recipe, lambda = 1e-4, threshold 0.1924). Those choices come from the full
experiment, which can be rerun with `python -B ml_attempts/05_feature_recovery/run.py`
(about 15 minutes) or, for all five attempts, `python -B ml_attempts/run_all.py`.

## Repository layout

| Path | Contents |
| --- | --- |
| [implementations.py](implementations.py) | The six required methods (GD/SGD least squares, normal equations, ridge, logistic, regularized logistic). |
| [run.py](run.py) | Reproduces the best submission. |
| [Exploratory Analysis/](<Exploratory Analysis/README.md>) | EDA of all fields and diagnostics of the six baseline methods. |
| [Mandatory Functions/](<Mandatory Functions/README.md>) | `implementations_augmented.py`: our reusable fitters (early stopping, Newton logistic), metrics and threshold search; unit tests for both modules. |
| [feature_processing/](feature_processing/README.md) | Training-only preprocessing: cleaning, encoding, scaling, engineered features, target encoding. |
| [ml_attempts/](ml_attempts/README.md) | The five modelling attempts, shared validation split, 3-fold confirmation and final refits. |
| `final_model/` | Frozen configuration of the submitted model. |
| `bible.md` | Dated team log of decisions and results. |

## Data preparation and feature generation

All statistics (medians, means, category levels) are fitted on training rows only and
then applied unchanged to validation, fold and test rows.

- **Removed:** identifiers, constant fields, exact duplicates, and fields that would leak
  the label or describe the survey rather than the person (e.g. cardiac rehabilitation,
  heart-related aspirin use, interview dates, survey weights).
- **Sparse fields** (over 95% missing) are replaced by a 0/1 "answered" indicator.
- **Redundant codings** of the same concept (age, height, weight/BMI, smoking, race,
  blood pressure, physical activity) are reduced to one chosen representation.
- **Survey codes** are decoded from the CDC 2015 codebook: "don't know"/"refused" (7/9,
  77/99) become missing, "none" (88) becomes 0, alcohol frequency becomes days per 30.
- **Numeric fields** are median-imputed and standardized; a missing indicator is added
  when at least 1% of values are missing.
- **Categorical fields** with up to 5 levels (8 for the recovered fields below) are
  one-hot encoded against a reference level, with separate "missing" and "other" columns;
  levels seen in fewer than max(100 rows, 0.1% of rows) are pooled into "other".
- **Engineered features:** a count of missing answers, age² and BMI², and seven
  age × condition interactions (fair/poor health, female, no diabetes, no high blood
  pressure, never smoker, high cholesterol).
- **Attempt 5 additions** (cumulative blocks F1-F4): decoded health-day, children and
  alcohol counts; diabetes, check-up, cholesterol-check, marital, education, employment
  and income categories; the interactions above; and prior stroke (`CVDSTRK3`), a
  comorbidity rather than part of the MICHD label.

## Validation protocol

1. **Shared split:** one stratified 80/20 split (seeds 42/43; 262,508 train / 65,627
   validation rows) is used by every attempt to choose hyperparameters. For each run the
   decision threshold is the one that maximizes validation F1, because the leaderboard
   ranks by F1 and the classes are imbalanced.
2. **3-fold confirmation:** the selected configuration of each attempt is re-evaluated on
   three stratified folds (seed 2026). In each fold, preprocessing is refitted from raw data
   on the training part; a 20% inner split picks the threshold (and the stopping point for
   gradient methods); the outer fold is only scored. Reported F1/accuracy are the means
   over outer folds.
3. **Final model:** refitted on all 328,135 rows; the threshold is the median of the three
   inner-fold thresholds. The test file is read only after all choices are frozen.

## Results (3-fold F1 / accuracy)

| Attempt | Model and features | F1 | Accuracy |
| --- | --- | --- | --- |
| 1 | GD L2 logistic, curated features (representation search) | 0.4125 | 0.860 |
| 2 | GD L2 logistic + out-of-fold target encoding | 0.4135 | 0.860 |
| 3 | Least squares by mini-batch SGD with early stopping | 0.4060 | 0.852 |
| 4 | Newton L2 logistic + target encoding | 0.4178 | 0.865 |
| 5 | Newton L2 logistic + codebook feature recovery + stroke (submitted) | **0.4279** | **0.867** |

Each attempt's folder README and `outputs/experiment_report.md` (generated) give the full
search, ablations and per-fold numbers.

## Tests

```text
python -B "Mandatory Functions/test_implementations.py"
python -B "Mandatory Functions/test_implementations_augmented.py"
python -B -m unittest feature_processing.test_preprocessing ml_attempts.test_experiments
python -B -m unittest discover -s "Exploratory Analysis" -p "test_*.py"
```
