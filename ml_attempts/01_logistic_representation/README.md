# Attempt 1: tuned logistic representations

Run `python -B ml_attempts/01_logistic_representation/run.py` from the repository
root. For separate selection then confirmation/export, add `--phase explore`
and later `--phase finish`. See [shared documentation](../README.md).

R0/A0 minimal is the benchmark to beat. R1/A2 and curated V1 each receive their
own staged lambda/gamma search using the existing augmented L2 logistic fitter
with no intercept penalty, zero starts and no class weights. Both default and
validation-selected thresholds are evaluated. Engineering is off by default.

`config.py` controls the four-lambda coarse search, two additional gammas and
optimization budgets. Stop reasons and actual/selected updates are recorded;
reaching max updates is not declared convergence. The chosen base recipe is
passed to Attempts 2 and 3, including any explicitly accepted engineering choice.

Start with [outputs/experiment_report.md](outputs/experiment_report.md).
Representation, lambda, gamma, threshold and optional-engineering tables accompany
the common run metrics. Three-fold confirmation precedes a complete-training
refit and local submission. Curated selectivity can remove predictive information;
no conceptual preference overrides measured validation evidence.
