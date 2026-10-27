# Attempt 2: selective target-encoded information recovery

Run `python -B ml_attempts/02_logistic_target_encoded/run.py` from the repository
root. A matching Attempt 1 selection is reused, or the prerequisite exploration
is run automatically. See [shared documentation](../README.md).

TE1 is the existing compact curated profile. TE2 starts with the actual winning
Attempt 1 base and adds a small, explicit block of training-supported categorical
fields that curated one-hot processing excluded for cardinality. Quarantine and
identifier exclusions are retained. Existing OOF encoders handle every training
row; full-training encoders transform only unseen validation/test data.

Use `config.py` to inspect the candidate priority list, maximum recovered count,
support requirements, smoothing default and competitiveness/material-gain rules.
The first test uses alpha 20 and a three-value lambda neighborhood around the
actual Attempt 1 winner. Only competitive variants receive a small alpha check.

[outputs/experiment_report.md](outputs/experiment_report.md) compares the actual
logistic benchmark with both TE variants, including dimensions, memory, time and
metric changes. Candidate groups are tested as a block; no unsupported individual
feature-importance claim is made. Tiny improvements do not automatically justify
encoding complexity. Confirmation and full-training local inference follow.
