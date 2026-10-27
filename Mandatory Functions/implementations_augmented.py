"""Experimental NumPy fitting tools; the mandatory course module is unchanged.

Inputs must already be prepared and finite. No intercept is added automatically.
Regularized defaults exempt column zero, assuming the caller supplied an intercept
there; set penalize_intercept=True when all coefficients should be penalized.
See README_augmented.md for weighting, checkpoint and selection conventions.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


__all__ = [
    'FitResult', 'fit_mse_gd', 'fit_mse_sgd', 'fit_least_squares',
    'fit_ridge_regression', 'fit_logistic_regression', 'fit_reg_logistic_regression',
    'linear_predict_scores', 'linear_predict', 'logistic_predict_proba',
    'logistic_predict', 'classification_metrics', 'find_best_threshold', 'make_class_weights',
]


@dataclass
class FitResult:
    """Selected weights, data loss and compact optimization diagnostics."""

    w: np.ndarray
    loss: float
    n_updates: int
    converged: bool
    stop_reason: str
    best_iteration: Optional[int]
    best_validation_loss: Optional[float]
    history: Optional[dict]


def _check_integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _check_scalar(value, name, minimum=0.0, strictly_positive=False):
    if not np.isscalar(value):
        raise ValueError(f'{name} must be a finite scalar')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{name} must be a finite scalar') from error
    if not np.isfinite(value) or value < minimum or strictly_positive and value <= 0:
        relation = '> 0' if strictly_positive else f'>= {minimum}'
        raise ValueError(f'{name} must be finite and {relation}')
    return value


def _validate_data(y, tx, *, logistic=False):
    y, tx = np.asarray(y, dtype=float), np.asarray(tx, dtype=float)
    if tx.ndim != 2:
        raise ValueError('tx must be a 2-D array of shape (N, D)')
    if y.ndim != 1:
        raise ValueError('y must be a 1-D array of shape (N,)')
    if len(y) != tx.shape[0]:
        raise ValueError('len(y) must equal tx.shape[0]')
    if not len(y) or not tx.shape[1]:
        raise ValueError('Training/validation data must contain rows and features')
    if not np.isfinite(y).all() or not np.isfinite(tx).all():
        raise ValueError('y and tx must be finite; prepare missing values before fitting')
    if logistic and not np.isin(y, [0, 1]).all():
        raise ValueError('Logistic labels must contain only 0 and 1')
    return y, tx


def _validate_weights(sample_weight, n):
    if sample_weight is None:
        return None
    weights = np.asarray(sample_weight, dtype=float)
    if weights.shape != (n,):
        raise ValueError('sample_weight must have shape (N,)')
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError('sample_weight must be finite, nonnegative and have positive sum')
    return weights


def _weight_probabilities(weights):
    # Scaling first avoids overflow when finite weights have an enormous sum.
    scaled = weights / np.max(weights)
    return scaled / np.sum(scaled)


def _compute_mse(y, tx, w, sample_weight=None):
    """Unregularized half-MSE, optionally normalized by total sample weight."""
    error = y - tx @ w
    if sample_weight is None:
        return float(0.5 * np.mean(error**2))
    return float(0.5 * (_weight_probabilities(sample_weight) @ (error**2)))


def _compute_mse_gradient(y, tx, w, sample_weight=None):
    error = y - tx @ w
    if sample_weight is None:
        return -(tx.T @ error) / len(y)
    return -(tx.T @ (_weight_probabilities(sample_weight) * error))


def _sigmoid(t):
    t = np.asarray(t, dtype=float)
    exp_negative_abs = np.exp(-np.abs(t))
    return np.where(t >= 0, 1.0 / (1.0 + exp_negative_abs),
                    exp_negative_abs / (1.0 + exp_negative_abs))


def _compute_logistic_loss(y, tx, w, sample_weight=None):
    scores = tx @ w
    losses = np.logaddexp(0.0, np.where(y == 1, -scores, scores))
    if sample_weight is None:
        return float(np.mean(losses))
    return float(_weight_probabilities(sample_weight) @ losses)


def _compute_logistic_gradient(y, tx, w, sample_weight=None):
    error = _sigmoid(tx @ w) - y
    if sample_weight is None:
        return tx.T @ error / len(y)
    return tx.T @ (_weight_probabilities(sample_weight) * error)


def _penalty_mask(d, penalize_intercept, intercept_index):
    index = _check_integer(intercept_index, 'intercept_index', 0)
    if index >= d:
        raise ValueError('intercept_index must be in range [0, D)')
    mask = np.ones(d)
    if not penalize_intercept:
        mask[index] = 0.0
    return mask


def _batch_indices(n, batch_size, seed, shuffle):
    """Yield full-data slices or successive local-RNG shuffled epoch batches."""
    if batch_size is None or batch_size >= n:
        while True:
            yield slice(None)
    rng = np.random.default_rng(seed)
    while True:
        indices = np.arange(n)
        if shuffle:
            rng.shuffle(indices)
        for start in range(0, n, batch_size):
            yield indices[start:start + batch_size]


def _learning_rate(gamma, update, schedule, decay):
    if schedule == 'sqrt':
        return gamma / np.sqrt(update)
    if schedule == 'inverse':
        return gamma / (1.0 + decay * update)
    return gamma


def _fit_iterative(
    y, tx, initial_w, max_iters, gamma, *, logistic, batch_size=None,
    sample_weight=None, lambda_=0.0, penalty_mask=None, seed=42, shuffle=True,
    tx_val=None, y_val=None, early_stopping=False, patience=20, tol=1e-6,
    gradient_tol=None, eval_every=10, lr_schedule='constant', lr_decay=0.0,
    return_history=False,
):
    """Shared functional update loop, checkpoint monitoring and result selection."""
    y, tx = _validate_data(y, tx, logistic=logistic)
    w = np.asarray(initial_w, dtype=float)
    if w.ndim != 1 or len(w) != tx.shape[1]:
        raise ValueError('initial_w must be a 1-D array of shape (D,)')
    if not np.isfinite(w).all():
        raise ValueError('initial_w must be finite')
    w = w.copy()
    max_iters = _check_integer(max_iters, 'max_iters', 0)
    eval_every = _check_integer(eval_every, 'eval_every', 1)
    patience = _check_integer(patience, 'patience', 1)
    gamma = _check_scalar(gamma, 'gamma', strictly_positive=True)
    tol = _check_scalar(tol, 'tol')
    lr_decay = _check_scalar(lr_decay, 'lr_decay')
    if gradient_tol is not None:
        gradient_tol = _check_scalar(gradient_tol, 'gradient_tol')
    if lr_schedule not in ('constant', 'sqrt', 'inverse'):
        raise ValueError('lr_schedule must be constant, sqrt or inverse')
    if batch_size is not None:
        batch_size = _check_integer(batch_size, 'batch_size', 1)
    if (tx_val is None) != (y_val is None):
        raise ValueError('Supply tx_val and y_val together')
    validation = tx_val is not None
    if early_stopping and not validation:
        raise ValueError('early_stopping requires a validation set')
    if validation:
        y_val, tx_val = _validate_data(y_val, tx_val, logistic=logistic)
        if tx_val.shape[1] != tx.shape[1]:
            raise ValueError('Validation and training feature counts must match')
    sample_weight = _validate_weights(sample_weight, len(y))
    probabilities = _weight_probabilities(sample_weight) if sample_weight is not None else None
    penalty_mask = np.zeros(tx.shape[1]) if penalty_mask is None else penalty_mask
    loss_function = _compute_logistic_loss if logistic else _compute_mse
    gradient_function = _compute_logistic_gradient if logistic else _compute_mse_gradient
    batches = _batch_indices(len(y), batch_size, seed, shuffle)
    history = {key: [] for key in ('update', 'train_loss', 'validation_loss',
                                   'learning_rate', 'weight_norm', 'gradient_norm')} if return_history else None
    best_loss, best_iteration, best_w = None, None, None
    stale, n_updates = 0, 0
    stop_reason = 'max_iters'
    converged = False

    # Checkpoint zero establishes a validation baseline and handles max_iters=0.
    # The final budget is also evaluated even when not divisible by eval_every.
    for update in range(max_iters + 1):
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            rate = 0.0 if update == 0 else _learning_rate(gamma, update, lr_schedule, lr_decay)
            if update:
                indices = next(batches)
                if probabilities is None:
                    gradient = gradient_function(y[indices], tx[indices], w)
                else:
                    batch_probabilities = probabilities[indices]
                    batch_mass = float(batch_probabilities.sum())
                    if batch_mass == 0:
                        gradient = np.zeros_like(w)
                    else:
                        gradient = gradient_function(y[indices], tx[indices], w, batch_probabilities)
                        # Uniform row traversal: an unbiased batch estimate of
                        # the globally normalized weighted objective, including
                        # batch_size=1. Local normalization alone cancels weights.
                        gradient *= len(y) * batch_mass / len(y[indices])
                gradient = gradient + 2 * lambda_ * penalty_mask * w
                w = w - rate * gradient
                n_updates = update
            if update % eval_every and update != max_iters:
                continue
            train_loss = loss_function(y, tx, w, sample_weight)
            validation_loss = loss_function(y_val, tx_val, w) if validation else np.nan
            gradient_norm = np.nan
            if return_history or gradient_tol is not None:
                full_gradient = gradient_function(y, tx, w, sample_weight) + 2 * lambda_ * penalty_mask * w
                gradient_norm = float(np.linalg.norm(full_gradient))
            finite = (np.isfinite(w).all() and np.isfinite(train_loss)
                      and (not validation or np.isfinite(validation_loss))
                      and (not (return_history or gradient_tol is not None) or np.isfinite(gradient_norm)))
            if history is not None:
                values = (update, train_loss, validation_loss, rate, float(np.linalg.norm(w)), gradient_norm)
                for key, value in zip(history, values):
                    history[key].append(value)
            if not finite:
                stop_reason = 'non_finite'
                break
            if validation:
                if best_loss is None or best_loss - validation_loss > tol:
                    best_loss, best_iteration, best_w = validation_loss, update, w.copy()
                    stale = 0
                elif update:
                    stale += 1
                if early_stopping and stale >= patience:
                    stop_reason, converged = 'early_stopping', True
                    break
            if gradient_tol is not None and gradient_norm <= gradient_tol:
                stop_reason, converged = 'tolerance', True
                break

    if early_stopping and best_w is not None and stop_reason != 'non_finite':
        w = best_w
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        final_loss = loss_function(y, tx, w, sample_weight)
    return FitResult(w, float(final_loss), n_updates, converged, stop_reason,
                     best_iteration, best_loss, history)


def fit_mse_gd(
    y, tx, initial_w, max_iters, gamma, *, sample_weight=None, tx_val=None,
    y_val=None, early_stopping=False, patience=20, tol=1e-6, gradient_tol=None,
    eval_every=10, lr_schedule='constant', lr_decay=0.0, return_history=False,
):
    """Full-batch half-MSE GD; default updates match the mandatory method."""
    return _fit_iterative(
        y, tx, initial_w, max_iters, gamma, logistic=False, sample_weight=sample_weight,
        tx_val=tx_val, y_val=y_val, early_stopping=early_stopping, patience=patience,
        tol=tol, gradient_tol=gradient_tol, eval_every=eval_every,
        lr_schedule=lr_schedule, lr_decay=lr_decay, return_history=return_history,
    )


def fit_mse_sgd(
    y, tx, initial_w, max_iters, gamma, *, batch_size=1, sample_weight=None,
    seed=42, shuffle=True, tx_val=None, y_val=None, early_stopping=False,
    patience=20, tol=1e-6, gradient_tol=None, eval_every=100,
    lr_schedule='constant', lr_decay=0.0, return_history=False,
):
    """Half-MSE with deterministic epoch-based SGD or mini-batches."""
    return _fit_iterative(
        y, tx, initial_w, max_iters, gamma, logistic=False, batch_size=batch_size,
        sample_weight=sample_weight, seed=seed, shuffle=shuffle, tx_val=tx_val,
        y_val=y_val, early_stopping=early_stopping, patience=patience, tol=tol,
        gradient_tol=gradient_tol, eval_every=eval_every, lr_schedule=lr_schedule,
        lr_decay=lr_decay, return_history=return_history,
    )


def fit_least_squares(y, tx, *, solver='solve'):
    """Normal equations or explicit pseudoinverse; there is no solver fallback."""
    y, tx = _validate_data(y, tx)
    if solver == 'solve':
        w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    elif solver == 'pinv':
        w = np.linalg.pinv(tx) @ y
    else:
        raise ValueError('solver must be solve or pinv')
    return FitResult(w, _compute_mse(y, tx, w), 0, True, 'closed_form', None, None, None)


def fit_ridge_regression(y, tx, lambda_, *, penalize_intercept=False, intercept_index=0):
    """Course-scaled ridge with optional intercept exemption; report data loss."""
    y, tx = _validate_data(y, tx)
    lambda_ = _check_scalar(lambda_, 'lambda_')
    mask = _penalty_mask(tx.shape[1], penalize_intercept, intercept_index)
    matrix = tx.T @ tx + 2 * len(y) * lambda_ * np.diag(mask)
    w = np.linalg.solve(matrix, tx.T @ y)
    return FitResult(w, _compute_mse(y, tx, w), 0, True, 'closed_form', None, None, None)


def fit_logistic_regression(
    y, tx, initial_w, max_iters, gamma, *, batch_size=None, sample_weight=None,
    seed=42, shuffle=True, tx_val=None, y_val=None, early_stopping=False,
    patience=20, tol=1e-6, gradient_tol=None, eval_every=10,
    lr_schedule='constant', lr_decay=0.0, return_history=False,
):
    """Binary {0,1} logistic GD with optional weighted epoch-based mini-batches."""
    return _fit_iterative(
        y, tx, initial_w, max_iters, gamma, logistic=True, batch_size=batch_size,
        sample_weight=sample_weight, seed=seed, shuffle=shuffle, tx_val=tx_val,
        y_val=y_val, early_stopping=early_stopping, patience=patience, tol=tol,
        gradient_tol=gradient_tol, eval_every=eval_every, lr_schedule=lr_schedule,
        lr_decay=lr_decay, return_history=return_history,
    )


def fit_reg_logistic_regression(
    y, tx, lambda_, initial_w, max_iters, gamma, *, batch_size=None,
    sample_weight=None, penalize_intercept=False, intercept_index=0, seed=42,
    shuffle=True, tx_val=None, y_val=None, early_stopping=False, patience=20,
    tol=1e-6, gradient_tol=None, eval_every=10, lr_schedule='constant',
    lr_decay=0.0, return_history=False,
):
    """L2 logistic fitting; returned loss excludes the masked L2 penalty."""
    lambda_ = _check_scalar(lambda_, 'lambda_')
    tx_array = np.asarray(tx)
    if tx_array.ndim != 2:
        raise ValueError('tx must be a 2-D array of shape (N, D)')
    mask = _penalty_mask(tx_array.shape[1], penalize_intercept, intercept_index)
    return _fit_iterative(
        y, tx, initial_w, max_iters, gamma, logistic=True, batch_size=batch_size,
        sample_weight=sample_weight, lambda_=lambda_, penalty_mask=mask,
        seed=seed, shuffle=shuffle, tx_val=tx_val, y_val=y_val,
        early_stopping=early_stopping, patience=patience, tol=tol,
        gradient_tol=gradient_tol, eval_every=eval_every, lr_schedule=lr_schedule,
        lr_decay=lr_decay, return_history=return_history,
    )


def make_class_weights(y, positive_weight=1.0, negative_weight=1.0):
    """Return a per-row weight for binary {0,1} labels."""
    y = np.asarray(y)
    if y.ndim != 1 or not np.isin(y, [0, 1]).all():
        raise ValueError('y must be a 1-D array containing only 0 and 1')
    positive_weight = _check_scalar(positive_weight, 'positive_weight')
    negative_weight = _check_scalar(negative_weight, 'negative_weight')
    weights = np.where(y == 1, positive_weight, negative_weight)
    _validate_weights(weights, len(y))
    return weights


def linear_predict_scores(tx, w):
    tx, w = np.asarray(tx, dtype=float), np.asarray(w, dtype=float)
    if tx.ndim != 2 or w.ndim != 1 or tx.shape[1] != len(w):
        raise ValueError('tx must have shape (N, D) and w must have shape (D,)')
    if not np.isfinite(tx).all() or not np.isfinite(w).all():
        raise ValueError('Prediction inputs must be finite')
    return tx @ w


def _validate_threshold(threshold):
    if not np.isscalar(threshold):
        raise ValueError('threshold must be a scalar')
    threshold = float(threshold)
    if np.isnan(threshold):
        raise ValueError('threshold must not be NaN')
    return threshold


def linear_predict(tx, w, threshold=0.0, positive_label=1, negative_label=-1):
    return np.where(linear_predict_scores(tx, w) >= _validate_threshold(threshold),
                    positive_label, negative_label)


def logistic_predict_proba(tx, w):
    return _sigmoid(linear_predict_scores(tx, w))


def logistic_predict(tx, w, threshold=0.5):
    return (logistic_predict_proba(tx, w) >= _validate_threshold(threshold)).astype(int)


def _validate_binary_labels(y, positive_label):
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError('Classification labels must be 1-D')
    if np.issubdtype(y.dtype, np.number) and not np.isfinite(y).all():
        raise ValueError('Classification labels must be finite')
    labels = np.unique(y)
    if len(labels) > 2 or len(labels) == 2 and positive_label not in labels:
        raise ValueError('Expected binary labels including positive_label when both classes are present')
    return y


def _metrics_from_counts(tp, tn, fp, fn):
    divide = lambda a, b: float(a / b) if b else 0.0
    recall, specificity = divide(tp, tp + fn), divide(tn, tn + fp)
    return {
        'accuracy': divide(tp + tn, tp + tn + fp + fn),
        'balanced_accuracy': 0.5 * (recall + specificity),
        'precision': divide(tp, tp + fp), 'recall': recall, 'specificity': specificity,
        'f1': divide(2 * tp, 2 * tp + fp + fn),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }


def classification_metrics(y_true, y_pred, *, positive_label=1):
    """Binary confusion metrics with zero for every undefined denominator."""
    y_true = _validate_binary_labels(y_true, positive_label)
    y_pred = _validate_binary_labels(y_pred, positive_label)
    if y_true.shape != y_pred.shape:
        raise ValueError('y_true and y_pred must have the same 1-D shape')
    _validate_binary_labels(np.concatenate([y_true, y_pred]), positive_label)
    positive, predicted = y_true == positive_label, y_pred == positive_label
    return _metrics_from_counts(
        np.sum(positive & predicted), np.sum(~positive & ~predicted),
        np.sum(~positive & predicted), np.sum(positive & ~predicted),
    )


def find_best_threshold(y_true, scores, *, metric='balanced_accuracy', thresholds=None, positive_label=1):
    """Choose a threshold on validation scores; never tune on training labels.

    Defaults to <=103 quantile/default/boundary candidates. Ties prefer higher
    balanced accuracy, then proximity to the inferred default, then lower threshold.
    """
    if metric not in ('balanced_accuracy', 'f1', 'recall', 'precision'):
        raise ValueError('metric must be balanced_accuracy, f1, recall or precision')
    y_true = _validate_binary_labels(y_true, positive_label)
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or scores.shape != y_true.shape or not len(scores):
        raise ValueError('scores and y_true must have matching, nonempty 1-D shapes')
    if not np.isfinite(scores).all():
        raise ValueError('scores must be finite')
    default = 0.5 if np.all((scores >= 0) & (scores <= 1)) else 0.0
    if thresholds is None:
        with np.errstate(over='ignore'):
            above_max = np.nextafter(scores.max(), np.inf)
        thresholds = np.r_[np.quantile(scores, np.linspace(0, 1, 101)), default, above_max]
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1 or not len(thresholds) or np.isnan(thresholds).any():
        raise ValueError('thresholds must be a nonempty 1-D array without NaN')
    order = np.argsort(scores, kind='stable')
    ordered_scores = scores[order]
    positives = (y_true[order] == positive_label).astype(int)
    prefix = np.r_[0, np.cumsum(positives)]
    total_positive = int(prefix[-1])
    best_key, best_threshold, best_metrics = None, None, None
    for threshold in np.unique(thresholds):
        below = int(np.searchsorted(ordered_scores, threshold, side='left'))
        fn = int(prefix[below])
        tp, tn = total_positive - fn, below - fn
        fp = len(scores) - below - tp
        metrics = _metrics_from_counts(tp, tn, fp, fn)
        key = (metrics[metric], metrics['balanced_accuracy'], -abs(threshold - default))
        if best_key is None or key > best_key:
            best_key, best_threshold, best_metrics = key, float(threshold), metrics
    return best_threshold, best_metrics
