"""Small staged searches, independently repeated for R0/R1/R2."""

CONFIG = {
    'attempt': '01_logistic_representation',
    # The leaderboard ranks by F1 (then accuracy): rank runs and pick every threshold by F1.
    'selection_metric': 'f1',
    'lambda_values': [0.001, 0.01, 0.1, 1.0],
    'gamma_values': [0.01, 0.03, 0.1], 'initial_gamma': 0.03,
    'batch_size': 1024, 'max_updates': 5000, 'eval_every': 100,
    'patience': 5, 'tol': 1e-5, 'seed': 42,
    'test_engineering': False, 'meaningful_gain': 0.002,
}
