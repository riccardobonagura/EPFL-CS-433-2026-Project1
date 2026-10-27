# Attempt 3: early-stopped MSE-SGD

Run `python -B ml_attempts/03_mse_sgd/run.py` from the repository root. This uses
the actual Attempt 1 winning recipe; missing prerequisite selection is generated
automatically. See [shared documentation](../README.md).

The existing augmented `fit_mse_sgd` receives {-1,+1} labels, local seeded epoch
batches, validation-MSE early stopping and an optional inverse schedule. Staged
gamma, batch and schedule tests avoid a Cartesian sweep. Defaults compare size
1/256/1024 and a few independently replayed update budgets.

Classification selection is external: each independent prefix returns its best
validation-MSE weights, then augmented metrics/threshold selection score those
weights for balanced accuracy, F1 and recall. Prefix replays share exact RNG and
schedule starts. Requested budget, performed updates and returned checkpoint are
separate columns; a low MSE is never assumed to mean good classification.

See [outputs/experiment_report.md](outputs/experiment_report.md), batch/gamma and
convergence tables, plus default and selected-threshold curves. The comparison is
against tuned logistic on the same base, with observed cost and data passes.
There is no automatic engineering block or extra representation search.
Early stopping is used in selection/confirmation; the all-training refit has no
validation rows and uses the previously classification-selected data-pass budget.
