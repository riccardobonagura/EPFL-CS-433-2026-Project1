"""Selective, support-filtered recovery; no quarantine or unknown numeric inputs."""

CONFIG = {
    "attempt": "02_logistic_target_encoded",
    # The leaderboard ranks by F1 (then accuracy): rank runs and pick every threshold by F1.
    "selection_metric": "f1",
    "alpha": 20.0,
    "competitive_alpha_values": [5.0, 100.0],
    "competitive_margin": 0.002,
    "meaningful_gain": 0.002,
    "target_folds": 5,
    "target_seed": 42,
    "max_recovered_features": 8,
    "minimum_observed": 1000,
    "candidate_priority": [
        "EDUCA",
        "INCOME2",
        "EMPLOY1",
        "_INCOMG",
        "_EDUCAG",
        "HHADULT",
        "CHILDREN",
        "CRGVREL1",
        "ADPLEASR",
        "ADDOWN",
        "ADSLEEP",
        "ADENERGY",
        "ADEAT1",
        "ADFAIL",
        "ADTHINK",
        "ADMOVE",
    ],
}
