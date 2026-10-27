"""Independent target-encoding experiment; creates Attempt 1 selection if needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml_attempts.orchestration import cli

if __name__ == '__main__':
    cli('02_logistic_target_encoded')
