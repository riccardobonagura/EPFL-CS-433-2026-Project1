"""Converged L2 logistic fits; one lambda grid shared by R2 and TE2 (matched control)."""

CONFIG = {
    'attempt': '04_logistic_newton',
    # The leaderboard ranks by F1 (then accuracy): rank runs and pick every threshold by F1.
    'selection_metric': 'f1',
    'lambda_values': [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    'max_iters': 50, 'gradient_tol': 1e-6,
    'recovery_alpha': 20.0,
}
