"""Staged MSE search with independent, seeded budget-prefix diagnostics."""

CONFIG = {
    'attempt': '03_mse_sgd', 'gamma_values': [0.001, 0.01, 0.03],
    'initial_batch_size': 256, 'batch_sizes': [1, 256, 1024],
    'max_updates': 4000, 'checkpoint_budgets': [500, 2000, 8000, 16000],
    'eval_every': 500, 'patience': 5, 'tol': 1e-5, 'seed': 42,
    'decaying_schedule': 'inverse', 'lr_decay': 0.0001,
}
