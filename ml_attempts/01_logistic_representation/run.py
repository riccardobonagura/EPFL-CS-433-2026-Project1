"""Independent entry point for the primary logistic representation experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml_attempts.orchestration import cli

if __name__ == "__main__":
    cli("01_logistic_representation")
