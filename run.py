"""Reproduce our best AIcrowd submission (F1 0.442): Attempt 5, block F4.

Usage (from the repository root, about 1-2 minutes):
    python run.py [--data-dir dataset] [--output submission.csv]

The data directory must contain the course files x_train.csv, y_train.csv and
x_test.csv. This script trains the final model on all 328,135 labelled rows and
predicts x_test.csv. It does not search hyperparameters: every choice (feature
recipe, Newton L2 logistic settings, decision threshold) is read from
final_model/frozen_configuration.json. That file is written by the full
experiment, `python -B ml_attempts/05_feature_recovery/run.py` (about 15 minutes:
lambda search on a validation split, then 3-fold confirmation that fixes the
threshold); `python -B ml_attempts/run_all.py` reruns all five attempts.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from feature_processing.io_utils import load_sampled_predictors, load_targets, load_unlabeled_csv
from ml_attempts.common import predict_scores
from ml_attempts.confirmation import _fit_full, write_submission
from ml_attempts.representations import fit_representation, transform_representation


def main():
    """Fit the frozen F4 model on all training rows and write the test predictions."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'dataset')
    parser.add_argument('--output', type=Path, default=ROOT / 'submission.csv')
    args = parser.parse_args()
    frozen = json.loads((ROOT / 'final_model/frozen_configuration.json').read_text(encoding='utf-8'))

    print('Loading training data...', flush=True)
    ids, y_pm1 = load_targets(args.data_dir / 'y_train.csv')
    names, x, _ = load_sampled_predictors(args.data_dir / 'x_train.csv', ids, np.arange(len(y_pm1)))

    # Preprocessing statistics (medians, scaling, category levels) come from training rows only.
    print(f'Fitting preprocessing and the Newton model on {len(x):,} rows...', flush=True)
    training, state, _ = fit_representation(x, y_pm1, names, frozen['recipe'])
    result, _ = _fit_full(training, y_pm1, frozen['params'])
    del training, x

    print('Predicting x_test.csv...', flush=True)
    raw_test, test_names, test_ids = load_unlabeled_csv(args.data_dir / 'x_test.csv')
    probabilities = predict_scores(frozen['params']['model'], transform_representation(raw_test, test_names, state), result.w)
    # The threshold maximizes F1 on held-out folds; the classes are imbalanced (8.8% positive).
    labels = np.where(probabilities >= frozen['threshold'], 1, -1)
    write_submission(args.output, test_ids, labels)
    print(f'Wrote {len(labels):,} predictions ({np.mean(labels == 1):.1%} positive) to {args.output}', flush=True)


if __name__ == '__main__':
    main()
