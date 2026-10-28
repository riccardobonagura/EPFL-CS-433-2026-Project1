# Attempt 5: codebook-decoded feature recovery and targeted interactions

Run `python -B ml_attempts/05_feature_recovery/run.py` from the repository root.
A matching Attempt 1 selection is reused, or the prerequisite exploration is run
automatically. See [shared documentation](../README.md).

The curated R2 base drops 33 numeric-looking fields as `unverified_numeric` and
41 fields as `unused_medium_cardinality`. These include diabetes status, checkup
and cholesterol-check recency, health-day counts, alcohol use and socioeconomic
fields. The only engineered terms tested before this attempt were the age/BMI squares.
This attempt adds four cumulative blocks, each built only from settings that
`feature_processing` already supports:

| Block | Change | Mechanism |
| --- | --- | --- |
| F0 | Attempt 1 base (R2 curated) | unchanged reference |
| F1 | + decoded numeric fields: `PHYSHLTH`, `MENTHLTH`, `POORHLTH`, `CHILDREN`, `ALCDAY5` (as drinking days per 30), `AVEDRNK2`, `DRNK3GE5`, `MAXDRNKS`, `DIABAGE2`, `_DRNKWEK` (log1p) | `special_code_map` (88 = none -> 0, 77/99 = missing) + `semantic_types: numeric` |
| F2 | + recovered categorical fields: `DIABETE3`, `CHECKUP1`, `CHOLCHK`, `MARITAL`, `EDUCA`, `EMPLOY1`, `INCOME2` | don't-know/refused codes -> missing, then one-hot with `onehot_limits` 8 |
| F3 | + seven age x condition interactions (fair/poor health, female, no diabetes, no high BP, never smoker, high cholesterol) and the age/BMI squares | `engineer` + `dummy_interactions` |
| F4 | + prior stroke (`CVDSTRK3`), released from the default quarantine | verified binary map (1 = yes, 2 = no; 7/9 = missing) |

Code meanings come from the BRFSS 2015 codebook value labels. In `x_train` they
appear as isolated, high-frequency values, as the codebook predicts; each mapping is listed in
[config.py](config.py). No field is decoded without an explicit mapping.

F4 is the only change to the default quarantine. The label (MICHD) means a reported heart attack,
angina or coronary heart disease. Those source questions are not in the data. Prior stroke is a
different, cerebrovascular diagnosis, asked of every respondent, and a recognized risk factor, like
the diabetes and hypertension fields already used. Fields that exist only because of a cardiac event
(cardiac rehabilitation `HAREHAB1`, which is 100% positive, and the aspirin questions) remain quarantined.
Stroke-rehab `STREHAB1` also remains quarantined.

All five blocks are fitted with converged Newton L2 logistic regression (the
Attempt 4 fitter) on one shared lambda grid, so a block's gain is not
confounded by an update cap or by per-block lambda retuning.
Runs are ranked, and every threshold is chosen, by F1 (`selection_metric` in
`config.py`), because the leaderboard ranks by F1 and then accuracy.
`matched_lambda_results.csv` lists F1, accuracy and BA per block, plus the
block-to-block F1 differences at each lambda. The overall winner goes through
the shared three-fold confirmation and the all-row refit, exactly as in
Attempt 4. Start with [outputs/experiment_report.md](outputs/experiment_report.md).
