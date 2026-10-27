# Experimental ML implementations

`implementations.py` remains the exact course/autograder API: six functions that
return `(w, loss)`. Its explanation remains in [README.md](README.md).

`implementations_augmented.py` is an independent experimental NumPy toolbox.
Its six `fit_*` functions return `FitResult` and add weighting, mini-batches,
validation monitoring, stopping rules and compact history. This module is not a
replacement for the mandatory submission file. It uses only NumPy and Python's
standard library and performs no training or file access at import time.

## Inputs and intercepts

Supply already prepared, finite arrays: `tx.shape == (N, D)`, `y.shape == (N,)`,
and `initial_w.shape == (D,)`. Column vectors are rejected. Inputs are not mutated.
Logistic training/validation labels must be 0 or 1. Convert original {-1,+1}
dataset labels in the calling pipeline, for example `(y_pm1 + 1) / 2`.

**No intercept column is added automatically.** If supplied, put it at
`intercept_index` (default zero). Augmented ridge and regularized logistic fitting
default to `penalize_intercept=False`. If there is no intercept, use
`penalize_intercept=True` to penalize every coefficient. Feature values are never
inspected to guess an intercept. The index must be in `[0, D)`.

## Fitting functions

| Function | Default behavior | Extra capabilities |
| --- | --- | --- |
| `fit_mse_gd` | Full-batch half-MSE GD | Sample weights, validation/stopping, schedules/history |
| `fit_mse_sgd` | Size-one epoch-based SGD | Other batch sizes, local seed/shuffling, same monitoring options |
| `fit_least_squares` | Normal equations with `solver="solve"` | Explicit `solver="pinv"` for a NumPy pseudoinverse solution |
| `fit_ridge_regression` | Course-scaled ridge, intercept exempt | Masked intercept penalty |
| `fit_logistic_regression` | Full-batch mean logistic-loss GD | Mini-batches, weights and all monitoring options |
| `fit_reg_logistic_regression` | L2 logistic GD, intercept exempt | Same additions plus masked intercept penalty |

Normal equations use `np.linalg.solve`. `solver="pinv"` explicitly uses
`np.linalg.pinv(tx) @ y`. There is no silent fallback, and `np.linalg.lstsq` is
never used. Singular direct solves propagate `np.linalg.LinAlgError`.

Course-equivalent modes:

- `fit_mse_gd` and full-batch `fit_logistic_regression`, with no weighting,
  stopping or learning-rate schedule, match their mandatory counterparts.
- Use `penalize_intercept=True` for mandatory-equivalent ridge and regularized
  logistic fitting. Ridge uses `X.T @ X + 2*N*lambda_*P`; logistic L2 gradient is
  `2*lambda_*(mask*w)`. All reported losses exclude the L2 penalty.
- Augmented SGD intentionally differs from mandatory SGD sampling: it walks
  through shuffled epochs rather than independently drawing random rows.

## Weighting and batches

Training sample weights must be finite, nonnegative, shape `(N,)`, and have a
positive sum. Weighted data losses are:

```text
half-MSE = 0.5 * sum(s * residual**2) / sum(s)
logistic = sum(s * binary_negative_log_likelihood) / sum(s)
```

Full gradients use the same total-weight normalization. Multiplying all weights
by a positive constant does not change fitting. Weight normalization is scaled
before summation to accommodate very large finite weights.

Mini-batches use the **global** normalized weights. For a batch B of size b, with
`q = s / sum(s)`, the data gradient is:

```text
(N / b) * X_B.T @ (q_B * residual_B)
```

For logistic fitting the residual is `sigmoid(X_B @ w) - y_B`; for MSE it is
`X_B @ w - y_B`. This gives an unbiased estimate of the full weighted gradient
under uniform random row sampling and preserves weighting for size-one SGD.
Renormalizing each single-row batch in isolation would cancel its class weight.
A batch with zero total weight has zero data gradient; its L2 update still applies.
Validation loss is always **unweighted**.

`make_class_weights(y_01, positive_weight=5, negative_weight=1)` creates per-row
weights for logistic experiments. The helper does not compute a weight ratio
from prevalence or silently convert labels.

`batch_size=None` or `batch_size>=N` means full-batch. Smaller batches traverse
each epoch sequentially, including its shorter final batch, and optionally
shuffle once at the beginning of each epoch using `np.random.default_rng(seed)`.
No global NumPy RNG state is read or modified. `max_iters` always counts gradient
updates, not epochs. `fit_mse_gd` remains full-batch only.

## Learning rates and stopping

Supported schedules, with the first update at `t=1`:

```text
constant: gamma
sqrt:     gamma / sqrt(t)
inverse:  gamma / (1 + lr_decay*t)
```

`gamma` must be finite and positive. Counts, decay, tolerances, labels and shapes
are validated with useful `ValueError` messages.

Evaluation checkpoints are **update 0**, every `eval_every` updates, and the final
budget if it is not a multiple of `eval_every`. Update 0 establishes a baseline
and permits `max_iters=0`. Every checkpoint checks weights and loss for numerical
failure. Validation observations never enter the gradients.

- Supply both `tx_val` and `y_val` to monitor validation. The best checkpoint is
  updated only for `best_loss - current_loss > tol`; `best_validation_loss` and
  `best_iteration` describe this strict, tolerance-qualified selection.
- `early_stopping=True` requires validation. `patience` counts stale **evaluation
  checkpoints**, not updates. It returns the saved best weights and recomputes
  training data loss at those weights. The initial weights can be best.
- With early-stopping mode enabled, the best weights are also selected if the
  iteration budget ends before patience is exhausted. In that case the reason
  remains `"max_iters"` and `converged=False`.
- With validation but early stopping disabled, return the last update weights;
  best-validation metadata remains available for inspection.
- `gradient_tol` checks the full training **objective** gradient, including L2
  and sample weights, at checkpoints only. It can stop immediately at update 0.
  Meeting the tolerance sets `stop_reason="tolerance"` and `converged=True`.
- Nonfinite optimization sets `stop_reason="non_finite"`, `converged=False` and
  retains the failing weights/loss for inspection. It does not hide the failure
  by restoring a finite validation checkpoint or repairing NaNs.

`FitResult.n_updates` counts work actually performed, even if returned weights
come from an earlier checkpoint. Closed-form fits have zero updates,
`converged=True`, `stop_reason="closed_form"`, and no validation/history fields.

`FitResult.loss` is training **data loss** at returned weights; with sample
weights it is the normalized weighted data loss. It never contains L2.

With `return_history=True`, history contains aligned lists named `update`,
`train_loss`, `validation_loss`, `learning_rate`, `weight_norm`, `gradient_norm`.
No weight vectors are retained in history. Validation entries are `NaN` without
validation; update-zero learning rate is 0. Gradient norms use the full regularized
training objective. The history's last row may differ from the selected best
weights. Full gradients are calculated at checkpoints only when requested by
history or gradient-tolerance monitoring. Otherwise `history=None`.

## Validation-based classification example

Run from this folder, with prepared arrays supplied by a separate pipeline.
`y_train` and `y_val` in this example are explicitly {0,1}; column zero is an
intercept provided by the caller.

```python
import numpy as np
from implementations_augmented import (
    fit_reg_logistic_regression,
    make_class_weights,
    logistic_predict_proba,
    logistic_predict,
    classification_metrics,
    find_best_threshold,
)

sample_weight = make_class_weights(y_train, positive_weight=5.0)
result = fit_reg_logistic_regression(
    y_train,
    tx_train,
    lambda_=0.1,
    initial_w=np.zeros(tx_train.shape[1]),
    max_iters=2000,
    gamma=0.05,
    batch_size=1024,
    sample_weight=sample_weight,
    penalize_intercept=False,
    tx_val=tx_val,
    y_val=y_val,
    early_stopping=True,
    patience=20,
    eval_every=10,
    seed=42,
    return_history=True,
)

val_prob = logistic_predict_proba(tx_val, result.w)
threshold, threshold_metrics = find_best_threshold(
    y_val, val_prob, metric="balanced_accuracy"
)
val_pred = logistic_predict(tx_val, result.w, threshold=threshold)
metrics = classification_metrics(y_val, val_pred)
```

This validation set selects both stopping and threshold. Its metrics are model
selection diagnostics, not an untouched final evaluation; assess the selected
model and fixed threshold on a separate holdout.

`linear_predict_scores(tx, w)` returns raw scores. `linear_predict` defaults to
positive label +1 at score >=0, otherwise -1, with configurable labels/threshold.
`logistic_predict_proba` uses a stable sigmoid; `logistic_predict` returns integer
0/1 at probability >=0.5 by default. Ties belong to the positive class.

`classification_metrics` supports binary numeric or string labels and returns
accuracy, balanced accuracy, precision, recall, specificity, F1, and `tp/tn/fp/fn`.
The designated positive label defaults to 1. Undefined ratios return 0.

`find_best_threshold` supports balanced accuracy, F1, recall or precision. Its
default deterministic grid has at most 103 candidates: score quantiles, the
inferred default, and a boundary above the maximum for an all-negative option.
Counts are computed by sorting scores once, preserving the `>=` tie convention.
Ranking ties prefer higher balanced accuracy, then proximity to 0.5 for scores
entirely in [0,1] or 0 otherwise, then the lower threshold. Explicit grids are
also accepted. Infinite thresholds are allowed for all-positive/all-negative
decisions; NaNs are rejected. Use this helper on **validation** scores, not the
training set or a final test set.

## Tests

From the repository root:

```text
python -B -m unittest discover -s "Mandatory Functions" -p "test_*.py" -v
```

The augmented suite checks equivalence with the immutable course functions,
independent weighted loss/gradient formulas, finite differences, weight scaling,
zero-weight mini-batches, epoch traversal, local-RNG reproducibility, intercept
exemption, validation isolation, selected weights, schedules, histories,
nonfinite stopping, threshold ties, confusion metrics, singular solvers and
input preservation. The original mandatory tests run alongside it.
