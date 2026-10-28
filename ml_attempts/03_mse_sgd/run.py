"""Independent MSE-SGD experiment on the actual Attempt 1 winning recipe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml_attempts.orchestration import cli

if __name__ == "__main__":
    cli("03_mse_sgd")
