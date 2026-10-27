# Attempt 4: converged Newton L2 logistic and a matched-lambda control

Run `python -B ml_attempts/04_logistic_newton/run.py` from the repository root.
A matching Attempt 1 selection is reused, or the prerequisite exploration is run
automatically. See [shared documentation](../README.md).

Every selected logistic run in Attempts 1 and 2 reached its 5,000-update cap, so
convergence was not established, and TE2's small gain over R2 came with a
different lambda. This attempt addresses both points with the augmented
`fit_reg_logistic_newton` fitter: full-batch Newton/IRLS steps with the exact
Hessian and a backtracking line search, on the same intercept-exempt L2 logistic
objective. Fits run until the full-objective gradient norm is below
`gradient_tol` (stop reason `tolerance`); there is no early stopping, so lambda
alone regularizes the model.

R2 (the actual Attempt 1 winning base) and TE2 (the same base plus selective OOF
target-encoded recovery, alpha 20) are both fitted at every lambda in
`config.py`. `matched_lambda_results.csv` reports TE2 minus R2 at equal lambda;
`solver_comparison.csv` compares the best Newton fit for each representation
with the Attempt 1/2 minibatch-GD selection.

The overall winner then goes through the shared three-fold confirmation and the
all-row refit. Confirmation and the final refit run Newton to the same tolerance
instead of rescaling a selected update budget. Start with
[outputs/experiment_report.md](outputs/experiment_report.md).
