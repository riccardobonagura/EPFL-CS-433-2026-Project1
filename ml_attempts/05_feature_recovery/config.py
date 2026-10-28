"""Cumulative codebook-decoded recovery blocks on R2; Newton fits on one shared lambda grid."""

# BRFSS 2015 codebook value labels. Each mapping was spot-checked against x_train:
# the 88/77/99/888/99900 codes appear as isolated high-frequency values.
DAYS = {88: 0.0, 77: "missing", 99: "missing"}
DRINKS = {77: "missing", 99: "missing"}
# ALCDAY5: 101-107 days per week, 201-230 days in past 30, 888 none; expressed as days per 30.
ALCOHOL_DAYS = {
    **{100 + k: round(k * 30 / 7, 6) for k in range(1, 8)},
    **{200 + k: float(k) for k in range(1, 31)},
    888: 0.0,
    777: "missing",
    999: "missing",
}

CONFIG = {
    "attempt": "05_feature_recovery",
    # The leaderboard ranks by F1 (then accuracy): rank runs and pick every threshold by F1.
    "selection_metric": "f1",
    "lambda_values": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
    "max_iters": 50,
    "gradient_tol": 1e-6,
    "meaningful_gain": 0.002,
    "decoded_numeric": {
        "PHYSHLTH": DAYS,
        "MENTHLTH": DAYS,
        "POORHLTH": DAYS,
        "CHILDREN": {88: 0.0, 99: "missing"},
        "ALCDAY5": ALCOHOL_DAYS,
        "AVEDRNK2": DRINKS,
        "DRNK3GE5": {88: 0.0, 77: "missing", 99: "missing"},
        "MAXDRNKS": DRINKS,
        "DIABAGE2": {98: "missing", 99: "missing"},
        "_DRNKWEK": {99900: "missing"},
    },
    "log1p_features": ["_DRNKWEK"],
    # 7/77 = don't know, 9/99 = refused; remaining levels are one-hot encoded.
    "recovered_categorical": {
        "DIABETE3": {7: "missing", 9: "missing"},
        "CHECKUP1": {7: "missing", 9: "missing"},
        "CHOLCHK": {7: "missing", 9: "missing"},
        "MARITAL": {9: "missing"},
        "EDUCA": {9: "missing"},
        "EMPLOY1": {9: "missing"},
        "INCOME2": {77: "missing", 99: "missing"},
    },
    "onehot_limit": 8,
    # F4: prior stroke is a comorbidity asked of every respondent, not part of the MICHD label
    # (heart attack / angina / coronary disease), so it leaves quarantine. Post-event fields
    # (cardiac rehab, aspirin for the heart) stay quarantined. 1 = yes, 2 = no, 7/9 = missing.
    "comorbidity_fields": ["CVDSTRK3"],
    "comorbidity_codes": {"CVDSTRK3": {7: "missing", 9: "missing"}},
    "comorbidity_binary_maps": {"CVDSTRK3": {1: 1, 2: 0}},
    # Reference levels are the lowest code, so e.g. DIABETE3__3 is "no diabetes".
    "interactions": {
        "age_x_fair_health": ["_AGE80__scaled", "GENHLTH__4"],
        "age_x_poor_health": ["_AGE80__scaled", "GENHLTH__5"],
        "age_x_female": ["_AGE80__scaled", "SEX__2"],
        "age_x_no_diabetes": ["_AGE80__scaled", "DIABETE3__3"],
        "age_x_no_high_bp": ["_AGE80__scaled", "BPHIGH4__3"],
        "age_x_never_smoker": ["_AGE80__scaled", "_SMOKER3__4"],
        "age_x_high_cholesterol": ["_AGE80__scaled", "_RFCHOL__2"],
    },
}
