"""Shared runner I/O and the already tested diagnostic sampling/splitting tools."""

import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "exploratory_analysis"))
from baseline_diagnostics import (
    load_targets,
    load_sampled_predictors,
    stratified_subsample,
    stratified_split,
)


def file_sha256(path):
    """Return the SHA-256 hex digest of a file, read in 1 MB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_training_partitions(data_dir, max_rows=0):
    """Load (optionally subsampled) training rows and split them 80/20 with fixed seeds."""
    ids, full_y = load_targets(Path(data_dir) / "y_train.csv")
    if max_rows < 0:
        raise ValueError("max_rows must be >=0 (0 means all rows)")
    sample_indices = stratified_subsample(full_y, max_rows or len(full_y), seed=42)
    names, x, integrity = load_sampled_predictors(
        Path(data_dir) / "x_train.csv", ids, sample_indices
    )
    y = full_y[sample_indices]
    train, validation = stratified_split(y, seed=43)
    sample_ids = np.asarray(ids)[sample_indices]
    return {
        "x_train": x[train],
        "x_validation": x[validation],
        "feature_names": names,
        "y_train": y[train],
        "y_validation": y[validation],
        "ids_train": sample_ids[train],
        "ids_validation": sample_ids[validation],
        "integrity": integrity,
        "sample_rows": len(x),
        "source_rows": len(full_y),
        "split_sha256": hashlib.sha256(
            sample_indices[train].astype("<i8").tobytes()
            + sample_indices[validation].astype("<i8").tobytes()
        ).hexdigest(),
    }


def load_unlabeled_csv(path):
    """Return (predictor matrix, names, IDs) in the input file's original order."""
    with open(path, "rb") as stream:
        n = sum(1 for line in stream if line.strip()) - 1
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if header.count("Id") != 1 or len(set(header)) != len(header):
            raise ValueError("Prediction CSV requires unique column names and one Id")
        id_index = header.index("Id")
        columns = [i for i in range(len(header)) if i != id_index]
        names = [header[i] for i in columns]
        x, ids = np.empty((n, len(columns))), []
        for index, row in enumerate(reader):
            if len(row) != len(header):
                raise ValueError("Prediction CSV record width mismatch")
            ids.append(row[id_index])
            x[index] = [
                np.nan if row[i].strip().lower() in ("", "nan", "na") else float(row[i])
                for i in columns
            ]
    if len(ids) != n or len(set(ids)) != n or not all(ids):
        raise ValueError("Prediction IDs must be present and unique")
    return x, names, np.asarray(ids)


def protect_modules():
    """Fingerprint the two ML implementation files so later changes can be detected."""
    return {
        str(path): file_sha256(path)
        for path in [
            PROJECT_DIR / "implementations.py",
            PROJECT_DIR / "implementations_improved/implementations_augmented.py",
        ]
    }


def check_modules(fingerprints):
    """Raise if any fingerprinted implementation file changed."""
    if any(file_sha256(path) != digest for path, digest in fingerprints.items()):
        raise RuntimeError("An ML implementation changed during execution")


def validate_output_directory(output, data_dir):
    """Create the output folder, refusing data, implementation or earlier analysis folders."""
    output = Path(output).resolve()
    protected = [
        Path(data_dir).resolve(),
        PROJECT_DIR / "implementations.py",
        PROJECT_DIR / "implementations_improved",
        PROJECT_DIR / "exploratory_analysis/eda_outputs",
        PROJECT_DIR / "exploratory_analysis/baseline_diagnostics",
    ]
    for path in protected:
        if output == path or output in path.parents or path in output.parents:
            raise ValueError(
                "Outputs must be separate from data, ML implementations and previous analysis outputs"
            )
    output.mkdir(parents=True, exist_ok=True)
    return output
