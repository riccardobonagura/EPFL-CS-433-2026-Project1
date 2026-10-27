"""Six Project 1 methods, adapted from the solutions in labs/ex02, ex04, ex05.

Vectors y and w have shapes (N,) and (D,); tx has shape (N, D).
All methods return (final weights, unregularized loss at those weights).
"""

import numpy as np


def _compute_mse(y, tx, w):
    """Half the mean squared residual, following lab 2's costs.py."""
    error = y - tx @ w
    return 0.5 * np.mean(error**2)


def _compute_mse_gradient(y, tx, w):
    """Gradient of half-MSE, following lab 2's compute_gradient."""
    error = y - tx @ w
    return -(tx.T @ error) / len(y)


def _sigmoid(t):
    """Lab 5's sigmoid, evaluated without exponentiating positive values."""
    exp_negative_abs = np.exp(-np.abs(t))
    return np.where(
        t >= 0,
        1.0 / (1.0 + exp_negative_abs),
        exp_negative_abs / (1.0 + exp_negative_abs),
    )


def _compute_logistic_loss(y, tx, w):
    """Mean negative log likelihood for binary {0, 1} labels (lab 5)."""
    scores = tx @ w
    # Equivalent to binary cross entropy; stable even for saturated sigmoid.
    signed_scores = np.where(y == 1, -scores, scores)
    return np.mean(np.logaddexp(0.0, signed_scores))


def _compute_logistic_gradient(y, tx, w):
    """Gradient of mean negative log likelihood, following lab 5."""
    return tx.T @ (_sigmoid(tx @ w) - y) / len(y)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Run exactly max_iters full-batch half-MSE gradient steps."""
    w = initial_w.copy()
    for _ in range(max_iters):
        w = w - gamma * _compute_mse_gradient(y, tx, w)
    return w, _compute_mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Run exactly max_iters half-MSE steps, each using one random sample."""
    w = initial_w.copy()
    for _ in range(max_iters):
        # Lab 2's batch_iter(batch_size=1, num_batches=1) draws this index.
        # Use NumPy's global RNG so callers can reproduce runs with a seed.
        index = np.random.randint(len(y), size=1)[0]
        y_batch = y[index : index + 1]
        tx_batch = tx[index : index + 1]
        w = w - gamma * _compute_mse_gradient(y_batch, tx_batch, w)
    return w, _compute_mse(y, tx, w)


def least_squares(y, tx):
    """Solve the normal equations as in labs 3/4; return weights and half-MSE."""
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, _compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Solve ridge normal equations; return weights and unregularized half-MSE."""
    # Lab 4: half-MSE + lambda_ * ||w||^2 regularizes every coefficient.
    penalty = 2 * len(y) * lambda_ * np.identity(tx.shape[1])
    w = np.linalg.solve(tx.T @ tx + penalty, tx.T @ y)
    return w, _compute_mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Run exactly max_iters GD steps on mean binary negative log likelihood."""
    w = initial_w.copy()
    for _ in range(max_iters):
        w = w - gamma * _compute_logistic_gradient(y, tx, w)
    return w, _compute_logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Train with an L2 penalty; return weights and unregularized logistic loss."""
    w = initial_w.copy()
    for _ in range(max_iters):
        # Lab 5: the derivative of lambda_ * ||w||^2 is 2 * lambda_ * w.
        gradient = _compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient
    return w, _compute_logistic_loss(y, tx, w)
