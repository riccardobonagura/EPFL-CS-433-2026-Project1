# Agentic Repository Guide

> **If an agent makes meaningful progress, changes architecture, completes a pending task, adds a new experiment, or discovers an important issue, it must update `agentic_readme.md` before finishing its work.**

> **Agents must log new design decisions in [bible.md](bible.md), appending entries at the end only. Never edit, delete or reorder earlier entries; append corrections or superseding decisions with references to the original entries.**

## Goal

Build reproducible coronary-disease classifiers, compare representations fairly, and produce local Project 1 submissions. This guide assumes familiarity with the project specification.

## Repository map

| Location / authority | Role |
| --- | --- |
| [bible.md](bible.md) | Detailed behind-the-scenes reasoning behind project design choices: evidence, tradeoffs, rejected alternatives and how decisions evolved. Read the relevant sections before changing a design. Preserve this project-wide reasoning history in append-only mode. |
| `dataset/` | Local CSV inputs; `architecture.md` records ordered fields/types. Preserve raw files and ID order. |
| [Mandatory Functions/README.md](<Mandatory Functions/README.md>) | `implementations.py`: six stable course/autograder functions. Keep this API separate from experimental extensions. |
| [Mandatory Functions/README_augmented.md](<Mandatory Functions/README_augmented.md>) | `implementations_augmented.py`: reusable fitters, stopping/history, weighting, metrics and threshold selection. Extend/reuse this layer for modeling. |
| [Exploratory Analysis/README.md](<Exploratory Analysis/README.md>) | Full-data EDA; separate [baseline diagnostics](<Exploratory Analysis/baseline_diagnostics_README.md>) exercise mandatory functions on a 100k sample. These are historical diagnostics. |
| [feature_processing/README.md](feature_processing/README.md) | Reusable fit/transform/state API in `preprocessing.py`; policies in `feature_config.py`, OOF encoding in `target_encoding.py`, engineering in `feature_engineering.py`. `evaluate_processing.py` owns preprocessing ablations. |
| [ml_attempts/README.md](ml_attempts/README.md) | Experiment entry points and phases. `data.py` owns split provenance, `representations.py` composes preprocessing, `searches.py` owns tuning, `confirmation.py` owns confirmation/refits. Each attempt's `config.py` is authoritative for its search settings. |
| `labs/` | Local course references, not the application architecture. |

Current attempt folders are **`01_logistic_representation`** (representation selection + L2 logistic), **`02_logistic_target_encoded`** (compact/selective target encoding + L2 logistic), and **`03_mse_sgd`** (winning curated representation + early-stopped MSE-SGD), under `ml_attempts/`.

Datasets, labs, generated reports/models/submissions, and saved split indices are Git-ignored. They exist locally but may be absent in another checkout. Inspect available artifacts before scheduling expensive regeneration.

## Key decisions / conventions

- Survey predictors are heavily categorical and missing. Numeric codes are not automatically continuous measurements or missing values; recode only through verified per-field policies. Missingness can carry signal. Bounded one-hot/rare grouping avoids encoding everything densely; provenance-sensitive features remain quarantined by default.
- Fit imputation, scaling, vocabulary and selection statistics on training rows only. Target-encoded training rows require **OOF `fit_transform`**; use fitted `transform` for unseen rows, never to reconstruct training encodings. Refit preprocessing inside every confirmation fold.
- Reuse shared preprocessing and augmented fitters; do not copy them into experiment scripts or rewrite mandatory functions. Preprocessing adds no intercept; the modeling wrapper supplies it. Logistic labels are 0/1; MSE uses -1/+1.
- Comparisons reuse the seeded stratified split: 262,508 train / 65,627 validation rows on current data. Saved indices address original CSV rows; hashes guard reuse. With roughly 8.8% positives, prioritize balanced accuracy (BA), F1 and recall. Report default and separately selected thresholds; never transplant an old run's threshold.

## Current status

### Completed

- Mandatory and augmented implementations are implemented and unit-tested. EDA and baseline diagnostics were executed; singular direct least-squares/ridge-zero baselines are recorded outcomes, not a reason to change the mandatory solver.
- Feature processing is implemented/tested, including state reload and leakage checks. Full-data processing and 100k-row ablations were executed. Older fixed-setting ablations favored minimal processing; subsequent retuning selected **R2 curated**, so added preprocessing is neither automatically beneficial nor disproven.
- All three experiment searches, three-fold finalist confirmations, and all-data refits were executed. Current confirmation mean BA: logistic **77.45%**, target encoding **77.57%**, MSE-SGD **77.10%**. The small encoding advantage is below the configured material-gain criterion; different regularization also limits attribution.
- Each attempt has local `outputs/selection.json`, `cv_metrics.csv`, `experiment_report.md`, saved preprocessing/weights and a 109,379-row `submission.csv`. Final refits use all 328,135 labeled rows. ID order, label domain and reload predictions were checked; experiment tests passed. Nothing was uploaded.
- The bible was audited against current code, full-data EDA recomputation and saved experiment scores/metadata; all 91 local tests passed. It distinguishes measured results from causal hypotheses, partition-dependent feature counts, five-fold OOF encoding, three-fold confirmation and frozen final refits. Small fold SD and TE2's slight gain do not establish absence of overfitting or an isolated encoding benefit.

### In progress

No active assignment or ownership ledger is recorded in the repository. Confirm teammate scope before assuming a workstream is available.

### Pending / next

- Choose the final project model and arrange external submission/report packaging; local exports already exist.
- Confirmation is conditional on finalists selected using overlapping exploratory data, not an independent or fully nested evaluation. Small differences do not establish a decisive winner.
- Optional follow-ups: matched-regularization encoding controls and optimization-budget checks. Several selected logistic searches reached their update cap; convergence is not established. These are open questions, not recorded active tasks.

## Working rules for agents

1. Read this guide, relevant reasoning in [bible.md](bible.md), the assigned folder's README/config, and `git status` before editing; preserve existing teammate changes.
2. Search for existing functionality, then extend shared infrastructure where appropriate. Coordinate shared-module edits; keep unrelated experiment folders untouched. Avoid stylistic refactors of working code.
3. Use saved metadata/reports to inspect results. Follow documented run phases deliberately: default experiment execution performs costly searches/refits. Run relevant tests from the affected layer; their commands are in its README.
4. Keep source/config changes separate from ignored run artifacts. Check split/config provenance before reusing results, and distinguish implemented, tested, executed and validated claims.

## Updating this file

Keep the map, decisions, status and pending work current. Replace stale statements rather than appending a changelog; record only information that changes the next agent's actions.

For each new design decision, also append a dated entry to [bible.md](bible.md) before finishing: state the decision, rationale, evidence or artifact links, and relevant alternatives/tradeoffs. Identify any earlier decision it corrects or supersedes. This guide stays a concise current overview; the bible retains the full chronological reasoning history, including decisions later reversed.
