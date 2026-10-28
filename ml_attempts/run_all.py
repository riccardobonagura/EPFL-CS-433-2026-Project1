"""Run selection, three-fold confirmation and all-data export for all attempts."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml_attempts.orchestration import cli

if __name__ == "__main__":
    cli()
