"""Reproducible Project 1 EDA using NumPy, the standard library, and plotting.

Run from any working directory: python "exploratory_analysis/eda.py".
No predictors or labels are recoded, and no models are trained.
"""

import argparse
import csv
import hashlib
import itertools
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

RANDOM_SEED = 433
EDA_SAMPLE_SIZE = 50_000
CATEGORICAL_SAMPLE_SIZE = 5_000
CSV_CHUNK_SIZE = 8_192
LOW_CARDINALITY = 10
MEDIUM_CARDINALITY = 30
DISCRETE_CARDINALITY = 100
NEAR_CONSTANT_FRACTION = 0.99
ALMOST_UNIQUE_FRACTION = 0.90
MIN_CATEGORY_SUPPORT = 100
MIN_CATEGORY_FRACTION = 0.001
MIN_GROUP_SUPPORT = 100
MISSING_RATE_DIFFERENCE = 0.03
CORRELATION_THRESHOLDS = (0.90, 0.95, 0.99)
MIN_PAIR_SUPPORT = 500
NEAR_DUPLICATE_AGREEMENT = 0.99
TAIL_ASYMMETRY_THRESHOLD = 3.0
SPECIAL_CODES = tuple(
    int(str(digit) * length) for digit in (7, 8, 9) for length in range(1, 6)
)

# Small, explicit name heuristics: these suggest review, never prove semantics.
IDENTIFIER_NAMES = {"ID", "SEQNO", "_PSU"}
DATE_NAMES = {"IDATE", "IMONTH", "IDAY", "IYEAR", "FMONTH", "FLSHTMY2", "HIVTSTD3"}
SURVEY_METADATA_NAMES = {
    "_STSTR",
    "_STRWT",
    "_RAWRAKE",
    "_WT2RAKE",
    "_CLLCPWT",
    "_DUALCOR",
    "_LLCPWT",
    "QSTVER",
    "QSTLANG",
    "MSCODE",
}
CATEGORICAL_TYPES = {
    "binary",
    "low-cardinality categorical",
    "medium-cardinality categorical",
}
NUMERIC_TYPES = {"ordinal-like / discrete numeric", "continuous-like numeric"}
STRATEGIES = (
    "L2 logistic regression with categorical encoding and missingness indicators",
    "Class-weighted L2 logistic regression with a small nonlinear feature set",
    "Ridge least-squares classifier with the same validated feature representation",
)


def load_numeric_csv(path):
    """Load numeric CSV chunks into one column-contiguous float64 array.

    The lab 6 loader uses the same genfromtxt primitive, but assumes a single
    combined CSV with string labels and drops its first two columns. That API
    does not fit the separate Project 1 files, so its layout logic is not reused.
    """
    with path.open("rb") as stream:
        n_rows = sum(1 for line in stream if line.strip()) - 1
    with path.open(newline="", encoding="utf-8-sig") as stream:
        header = next(csv.reader(stream))
        if not header or len(set(header)) != len(header):
            raise ValueError(f"{path.name}: empty or duplicate column names")
        if n_rows <= 0:
            raise ValueError(f"{path.name}: no data rows")
        data = np.empty((n_rows, len(header)), dtype=np.float64, order="F")
        offset = 0
        while True:
            lines = list(itertools.islice(stream, CSV_CHUNK_SIZE))
            if not lines:
                break
            if any(not line.strip() for line in lines):
                raise ValueError(f"{path.name}: blank records are not supported")
            block = np.genfromtxt(
                lines,
                delimiter=",",
                dtype=np.float64,
                ndmin=2,
                missing_values={None: ("", "NaN", "nan", "NA")},
                filling_values=np.nan,
                loose=False,
                invalid_raise=True,
            )
            if block.shape[1] != len(header):
                raise ValueError(f"{path.name}: record width differs from its header")
            if np.isinf(block).any():
                raise ValueError(f"{path.name}: infinite numeric value detected")
            data[offset : offset + len(block)] = block
            offset += len(block)
    if offset != n_rows:
        raise ValueError(f"{path.name}: row count changed during loading")
    return header, data


def validate_ids_and_shapes(x_header, x, y_header, targets):
    """Verify IDs before any target association; fail rather than silently join."""
    if "Id" not in x_header or set(y_header) != {"Id", "_MICHD"}:
        raise ValueError("Expected Id in X and exactly Id, _MICHD in y")
    x_ids = x[:, x_header.index("Id")]
    y_ids = targets[:, y_header.index("Id")]
    integrity = {
        "x_rows": len(x),
        "predictors": len(x_header) - 1,
        "y_rows": len(targets),
    }
    for prefix, ids in [("x", x_ids), ("y", y_ids)]:
        finite = np.isfinite(ids)
        integrity[f"{prefix}_missing_ids"] = int((~finite).sum())
        valid_ids = ids[finite]
        integrity[f"{prefix}_duplicate_ids"] = int(
            len(valid_ids) - len(np.unique(valid_ids))
        )
        if np.any(valid_ids != np.floor(valid_ids)) or np.any(
            np.abs(valid_ids) >= 2**53
        ):
            raise ValueError(
                "This numeric loader requires integer IDs with magnitude < 2**53"
            )
    integrity["same_id_set"] = bool(np.array_equal(np.unique(x_ids), np.unique(y_ids)))
    integrity["exact_id_order"] = bool(np.array_equal(x_ids, y_ids))
    if (
        any(
            integrity[f"{prefix}_{issue}_ids"]
            for prefix in ("x", "y")
            for issue in ("missing", "duplicate")
        )
        or not integrity["same_id_set"]
        or not integrity["exact_id_order"]
    ):
        raise ValueError("ID integrity/alignment failure: " + json.dumps(integrity))
    y = targets[:, y_header.index("_MICHD")]
    if not np.array_equal(np.unique(y), [-1, 1]):
        raise ValueError("Expected both target classes, with labels exactly {-1, +1}")
    return integrity, y


def infer_candidate_type(name, unique_count, unique_fraction, integer_like, observed):
    """Apply ordered candidate-type heuristics documented in the report."""
    if name.upper() in IDENTIFIER_NAMES:
        return "identifier", "explicit identifier/sequence name heuristic"
    if not observed:
        return "unknown", "no observed values"
    if unique_count == 1:
        return "constant", "one distinct observed value"
    if name.upper() in DATE_NAMES:
        return "date-like", "explicit date/component name heuristic"
    if unique_count == 2:
        return "binary", "two distinct observed values; meanings unverified"
    if unique_count <= LOW_CARDINALITY:
        return (
            "low-cardinality categorical",
            f"3..{LOW_CARDINALITY} distinct observed values",
        )
    if unique_count <= MEDIUM_CARDINALITY:
        return (
            "medium-cardinality categorical",
            f"{LOW_CARDINALITY + 1}..{MEDIUM_CARDINALITY} distinct values",
        )
    if integer_like and unique_fraction >= ALMOST_UNIQUE_FRACTION:
        return (
            "high-cardinality suspicious",
            "integer-like and almost unique among observed rows",
        )
    if integer_like and unique_count <= DISCRETE_CARDINALITY:
        return (
            "ordinal-like / discrete numeric",
            "integer-like, 31..100 values; ordering unverified",
        )
    return (
        "continuous-like numeric",
        "remaining numeric field; semantic interpretation still unverified",
    )


def semantic_review_reasons(name, kind, unique_fraction):
    """Reasons a field needs a manual leakage/metadata review (empty list if none)."""
    reasons = []
    if kind in {"identifier", "high-cardinality suspicious", "date-like"}:
        reasons.append(
            "record/date/identity information: review provenance and split dependence"
        )
    if name in SURVEY_METADATA_NAMES:
        reasons.append(
            "survey weight/design/administration name: verify availability and intended use"
        )
    elif name.startswith("_"):
        reasons.append(
            "derived/survey-processing name: check derivation and input provenance"
        )
    if name.startswith(
        ("CVD", "HAREHAB", "STREHAB", "RDUCHART", "RDUCSTRK", "CVDASPRN")
    ):
        reasons.append(
            "cardiovascular-related name: check relationship and timing relative to _MICHD"
        )
    if unique_fraction >= ALMOST_UNIQUE_FRACTION:
        reasons.append(
            "almost unique values: check measurement versus identity/design metadata"
        )
    return reasons


def _rate(positive_count, count):
    """positive_count / count, or NaN when count is zero."""
    return float(positive_count / count) if count else np.nan


def compute_feature_summaries(header, data, y):
    """Compute exact full-data summaries with vectorized work within each column."""
    positive = y == 1
    rows, missing_rows, codes, categories, associations, reviews = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    row_missing = np.zeros(len(y), dtype=np.int32)
    duplicate_buckets, pattern_buckets = defaultdict(list), defaultdict(list)
    feature_indices = [i for i, name in enumerate(header) if name != "Id"]
    for feature_index, column_index in enumerate(feature_indices):
        name = header[column_index]
        column = data[:, column_index]
        present = np.isfinite(column)
        missing = ~present
        row_missing += missing
        values = column[present]
        observed_positive = positive[present]
        unique, inverse, counts = np.unique(
            values, return_inverse=True, return_counts=True
        )
        n, k = len(values), len(unique)
        unique_fraction = k / n if n else np.nan
        integer_like = bool(n and np.all(values == np.floor(values)))
        kind, reason = infer_candidate_type(name, k, unique_fraction, integer_like, n)
        mode_index = int(np.argmax(counts)) if k else None
        mode_fraction = float(counts[mode_index] / n) if n else np.nan
        percentiles = (
            np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
            if n
            else np.full(7, np.nan)
        )
        tail_asymmetry = (
            float((percentiles[5] - percentiles[3]) / (percentiles[3] - percentiles[1]))
            if n and percentiles[3] > percentiles[1]
            else np.nan
        )
        record = {
            "feature_name": name,
            "column_index": column_index,
            "feature_index": feature_index,
            "non_missing_count": n,
            "missing_count": int(missing.sum()),
            "missing_fraction": float(missing.mean()),
            "unique_count": k,
            "unique_fraction": unique_fraction,
            "minimum": float(values.min()) if n else np.nan,
            "maximum": float(values.max()) if n else np.nan,
            "mean": float(values.mean()) if n else np.nan,
            "std": float(values.std()) if n else np.nan,
            **dict(
                zip(
                    ("p01", "p05", "p25", "median", "p75", "p95", "p99"),
                    map(float, percentiles),
                )
            ),
            "candidate_feature_type": kind,
            "type_heuristic": reason,
            "integer_like": integer_like,
            "constant_flag": k == 1,
            "near_constant_flag": bool(
                k > 1 and mode_fraction > NEAR_CONSTANT_FRACTION
            ),
            "most_common_value": float(unique[mode_index]) if k else np.nan,
            "most_common_frequency": int(counts[mode_index]) if k else 0,
            "most_common_fraction": mode_fraction,
            "upper_lower_tail_ratio": tail_asymmetry,
        }
        code_count = 0
        for value in SPECIAL_CODES:
            location = np.searchsorted(unique, value)
            if (
                location < k
                and unique[location] == value
                and counts[location] >= MIN_CATEGORY_SUPPORT
            ):
                count = counts[location]
                codes.append(
                    {
                        "feature_name": name,
                        "candidate_code": float(value),
                        "frequency": int(count),
                        "fraction_of_observed": float(count / n),
                    }
                )
                code_count += 1
        record["possible_special_code_count"] = code_count
        reasons = semantic_review_reasons(name, kind, unique_fraction)
        if reasons:
            reviews.append({"feature_name": name, "reasons": "; ".join(reasons)})
        removal_reasons = []
        if kind in {"identifier", "high-cardinality suspicious", "constant", "unknown"}:
            removal_reasons.append(kind)
        if record["near_constant_flag"]:
            removal_reasons.append("near constant among observed values")
        if name in SURVEY_METADATA_NAMES or kind == "date-like":
            removal_reasons.append("survey metadata/date: review before use")
        if record["missing_fraction"] > 0.95:
            removal_reasons.append("more than 95% missing")
        flags = []
        if removal_reasons:
            flags.append("candidate_drop")
        if kind == "binary":
            flags.append("leave_binary_after_mapping_to_0_1")
        elif kind in CATEGORICAL_TYPES or kind == "ordinal-like / discrete numeric":
            flags.append("categorical_encode_or_transform")
        if kind in NUMERIC_TYPES:
            flags.append("standardize_after_semantic_review")
        if code_count:
            flags.append("investigate_special_codes")
        if missing.any():
            flags.append("missing_value_treatment_needed")
        record["candidate_for_removal"] = bool(removal_reasons)
        record["removal_review_reason"] = "; ".join(removal_reasons)
        record["preprocessing_flags"] = "; ".join(flags)
        record["semantic_review"] = bool(reasons)
        rows.append(record)

        missing_n, present_n = int(missing.sum()), n
        missing_p, present_p = int(positive[missing].sum()), int(
            observed_positive.sum()
        )
        missing_rate, present_rate = _rate(missing_p, missing_n), _rate(
            present_p, present_n
        )
        rate_difference = missing_rate - present_rate
        support = missing_n >= MIN_GROUP_SUPPORT and present_n >= MIN_GROUP_SUPPORT
        missing_phi = 0.0
        if support:
            p, m = positive.mean(), missing.mean()
            missing_phi = abs(rate_difference) * np.sqrt(m * (1 - m) / (p * (1 - p)))
        missing_rows.append(
            {
                "feature_name": name,
                "missing_fraction": float(missing.mean()),
                "missing_count": missing_n,
                "present_count": present_n,
                "positive_count_when_missing": missing_p,
                "positive_count_when_present": present_p,
                "positive_rate_when_missing": missing_rate,
                "positive_rate_when_present": present_rate,
                "difference": rate_difference,
                "absolute_difference": abs(rate_difference),
                "sufficient_support": support,
                "strong_missingness_flag": bool(
                    support and abs(rate_difference) >= MISSING_RATE_DIFFERENCE
                ),
            }
        )

        group_positive = np.bincount(
            inverse, weights=observed_positive.astype(float), minlength=k
        )
        category_support = max(
            MIN_CATEGORY_SUPPORT, int(np.ceil(MIN_CATEGORY_FRACTION * n))
        )
        eta = 0.0
        if kind in CATEGORICAL_TYPES:
            supported = counts >= category_support
            if supported.sum() >= 2:
                weights = counts[supported] / counts[supported].sum()
                rates = group_positive[supported] / counts[supported]
                supported_rate = float(weights @ rates)
                if 0 < supported_rate < 1:
                    eta = float(
                        np.sqrt(
                            weights
                            @ (rates - supported_rate) ** 2
                            / (supported_rate * (1 - supported_rate))
                        )
                    )
            for value, count, p_count in zip(unique, counts, group_positive):
                categories.append(
                    {
                        "feature_name": name,
                        "category": float(value),
                        "count": int(count),
                        "positive_count": int(p_count),
                        "positive_rate": float(p_count / count),
                        "minimum_support": category_support,
                        "supported": bool(count >= category_support),
                    }
                )
        pos_values, neg_values = values[observed_positive], values[~observed_positive]
        smd, point_biserial = np.nan, np.nan
        if (
            kind in NUMERIC_TYPES
            and min(len(pos_values), len(neg_values)) >= MIN_GROUP_SUPPORT
        ):
            pooled_std = np.sqrt((pos_values.var() + neg_values.var()) / 2)
            difference = pos_values.mean() - neg_values.mean()
            smd = float(difference / pooled_std) if pooled_std else np.nan
            p = observed_positive.mean()
            point_biserial = (
                float(difference * np.sqrt(p * (1 - p)) / values.std())
                if values.std()
                else np.nan
            )
        eligible = (
            kind in CATEGORICAL_TYPES | NUMERIC_TYPES | {"constant"}
            and name not in SURVEY_METADATA_NAMES
        )
        numeric_score = abs(point_biserial) if np.isfinite(point_biserial) else 0.0
        components = {
            "categorical_eta": eta,
            "numeric_abs_r": numeric_score,
            "missingness_abs_phi": float(missing_phi),
        }
        score = max(components.values()) if eligible else np.nan
        associations.append(
            {
                "feature_name": name,
                "candidate_feature_type": kind,
                "ranking_eligible": eligible,
                "association_score": score,
                "score_component": max(components, key=components.get),
                **components,
                "standardized_mean_difference": smd,
                "point_biserial_r": point_biserial,
                "positive_observed_count": len(pos_values),
                "negative_observed_count": len(neg_values),
                "positive_mean": (
                    float(pos_values.mean()) if len(pos_values) else np.nan
                ),
                "negative_mean": (
                    float(neg_values.mean()) if len(neg_values) else np.nan
                ),
                "positive_median": (
                    float(np.median(pos_values)) if len(pos_values) else np.nan
                ),
                "negative_median": (
                    float(np.median(neg_values)) if len(neg_values) else np.nan
                ),
            }
        )
        canonical = column.copy()
        canonical[np.isnan(canonical)] = np.nan
        canonical[canonical == 0] = 0  # Normalize signed zero before hashing.
        digest = hashlib.blake2b(canonical.tobytes(), digest_size=16).digest()
        duplicate_buckets[digest].append(column_index)
        if missing.any() and present.any():
            digest = hashlib.blake2b(
                np.packbits(missing).tobytes(), digest_size=16
            ).digest()
            pattern_buckets[digest].append(column_index)

    duplicates = []
    for indices in duplicate_buckets.values():
        for left, right in itertools.combinations(indices, 2):
            if np.array_equal(data[:, left], data[:, right], equal_nan=True):
                duplicates.append(
                    {
                        "feature_a": header[left],
                        "feature_b": header[right],
                        "scope": "full dataset, including matching NaN positions",
                    }
                )
    patterns = []
    for indices in pattern_buckets.values():
        if len(indices) > 1:
            first = indices[0]
            # Verify hash matches before calling patterns identical.
            group = [
                i
                for i in indices
                if np.array_equal(np.isnan(data[:, first]), np.isnan(data[:, i]))
            ]
            if len(group) > 1:
                patterns.append(
                    {
                        "feature_count": len(group),
                        "features": "; ".join(header[i] for i in group),
                        "missing_fraction": float(np.isnan(data[:, first]).mean()),
                    }
                )
    return (
        rows,
        missing_rows,
        codes,
        categories,
        associations,
        reviews,
        row_missing,
        duplicates,
        patterns,
    )


def pairwise_complete_correlation(sample, min_support=MIN_PAIR_SUPPORT):
    """Pearson correlation using each pair's observed intersection, without imputation."""
    mask = np.isfinite(sample)
    observed = mask.sum(axis=0)
    center = np.divide(
        np.nansum(sample, axis=0),
        observed,
        out=np.zeros(sample.shape[1]),
        where=observed > 0,
    )
    z = np.where(mask, sample - center, 0.0)
    m = mask.astype(float)
    counts = m.T @ m
    safe_counts = np.maximum(counts, 1)
    sums = z.T @ m
    squares = (z * z).T @ m
    covariance = z.T @ z - sums * sums.T / safe_counts
    variances = np.maximum(squares - sums * sums / safe_counts, 0)
    denominator = np.sqrt(variances * variances.T)
    correlation = np.divide(
        covariance,
        denominator,
        out=np.full_like(covariance, np.nan),
        where=denominator > 0,
    )
    correlation[counts < min_support] = np.nan
    return np.clip(correlation, -1, 1), counts.astype(int)


def analyze_redundancy(data, summaries, sample_indices):
    """Find highly correlated numeric field pairs on a row sample."""
    numeric = [r for r in summaries if r["candidate_feature_type"] in NUMERIC_TYPES]
    sample = data[np.ix_(sample_indices, [r["column_index"] for r in numeric])]
    correlation, counts = pairwise_complete_correlation(sample)
    pairs = []
    for i, j in itertools.combinations(range(len(numeric)), 2):
        value = correlation[i, j]
        if np.isfinite(value) and abs(value) > CORRELATION_THRESHOLDS[0]:
            pairs.append(
                {
                    "feature_a": numeric[i]["feature_name"],
                    "feature_b": numeric[j]["feature_name"],
                    "correlation": float(value),
                    "absolute_correlation": abs(float(value)),
                    "pair_observed_count": int(counts[i, j]),
                    "sample_rows": len(sample_indices),
                }
            )
    pairs.sort(key=lambda r: -r["absolute_correlation"])
    categorical = [
        r for r in summaries if r["candidate_feature_type"] in CATEGORICAL_TYPES
    ]
    cat_indices = sample_indices[:CATEGORICAL_SAMPLE_SIZE]
    cat_sample = data[np.ix_(cat_indices, [r["column_index"] for r in categorical])]
    agreements = []
    for i, j in itertools.combinations(range(len(categorical)), 2):
        left, right = cat_sample[:, i], cat_sample[:, j]
        present = np.isfinite(left) & np.isfinite(right)
        support = int(present.sum())
        if support < MIN_PAIR_SUPPORT:
            continue
        agreement = float(np.mean(left[present] == right[present]))
        # Exclude trivial agreement of constant observed intersections.
        if (
            agreement >= NEAR_DUPLICATE_AGREEMENT
            and len(np.unique(left[present])) > 1
            and len(np.unique(right[present])) > 1
        ):
            agreements.append(
                {
                    "feature_a": categorical[i]["feature_name"],
                    "feature_b": categorical[j]["feature_name"],
                    "observed_agreement": agreement,
                    "pair_observed_count": support,
                    "missingness_agreement": float(
                        np.mean(np.isnan(left) == np.isnan(right))
                    ),
                    "sample_rows": len(cat_indices),
                }
            )
    agreements.sort(key=lambda r: -r["observed_agreement"])
    return pairs, agreements, numeric, correlation


def write_csv(path, rows, empty_header):
    """Write dict rows as CSV (header only if empty)."""
    keys = list(rows[0]) if rows else empty_header
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        ""
                        if isinstance(value, (float, np.floating))
                        and not np.isfinite(value)
                        else value
                    )
                    for key, value in row.items()
                }
            )


def missingness_bands(summaries):
    """Count fields per missing-fraction band (0%, up to 1%, ..., over 95%)."""
    fractions = np.array([r["missing_fraction"] for r in summaries])
    edges = [0, 0.01, 0.1, 0.3, 0.5, 0.8, 0.95, 1]
    rows = [("0%", int((fractions == 0).sum()))]
    labels = [
        "(0, 1%]",
        "(1%, 10%]",
        "(10%, 30%]",
        "(30%, 50%]",
        "(50%, 80%]",
        "(80%, 95%]",
        ">95%",
    ]
    for label, low, high in zip(labels, edges, edges[1:]):
        rows.append((label, int(((fractions > low) & (fractions <= high)).sum())))
    return rows


def _format(value):
    """Format a value for a Markdown table cell."""
    if isinstance(value, (float, np.floating)):
        return f"{value:.4g}" if np.isfinite(value) else "not available"
    return str(value).replace("|", "/").replace("\n", " ")


def markdown_table(rows, columns, limit=12):
    """Render up to limit rows as a Markdown table."""
    lines = [
        "| " + " | ".join(title for _, title in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows[:limit]:
        lines.append("| " + " | ".join(_format(row[key]) for key, _ in columns) + " |")
    if not rows:
        return "None met the documented criteria.\n"
    return "\n".join(lines) + "\n"


def generate_figures(
    output,
    summaries,
    y,
    row_missing,
    ranking,
    numeric,
    correlation,
    data,
    sample_indices,
):
    """Save the EDA matplotlib figures (missingness, target rates, correlations)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        """Save one figure as PNG and close it."""
        fig.tight_layout()
        fig.savefig(
            figures / name, dpi=140, metadata={"Software": "Project 1 NumPy EDA"}
        )
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    counts = [int((y == label).sum()) for label in (-1, 1)]
    bars = ax.bar(
        ["Negative (-1)", "Positive (+1)"], counts, color=["#3b6c8d", "#d76e4d"]
    )
    ax.bar_label(bars, labels=[f"{n:,} ({n / len(y):.2%})" for n in counts], padding=3)
    ax.set(
        title="Target distribution (full training set)",
        xlabel="Target class",
        ylabel="People (count)",
        ylim=(0, max(counts) * 1.15),
    )
    save(fig, "01_target_distribution.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(
        [r["missing_fraction"] * 100 for r in summaries],
        bins=np.linspace(0, 100, 21),
        color="#3b6c8d",
    )
    ax.set(
        title="Missingness across predictors (full training set)",
        xlabel="Missing values per feature (%)",
        ylabel="Predictors (count)",
    )
    save(fig, "02_missingness_distribution.png")

    top_missing = sorted(summaries, key=lambda r: -r["missing_fraction"])[:40][::-1]
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.barh(
        [r["feature_name"] for r in top_missing],
        [r["missing_fraction"] * 100 for r in top_missing],
        color="#3b6c8d",
    )
    ax.set(
        title="40 predictors with highest missingness (full training set)",
        xlabel="Missing values (%)",
        ylabel="Predictor",
        xlim=(0, 100),
    )
    save(fig, "03_missingness_by_feature.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    for label, color in [(-1, "#3b6c8d"), (1, "#d76e4d")]:
        ax.hist(
            row_missing[y == label] / len(summaries) * 100,
            bins=np.linspace(0, 100, 41),
            weights=np.full(int((y == label).sum()), 1 / (y == label).sum()),
            histtype="step",
            linewidth=2,
            label=f"Class {label:+d}",
            color=color,
        )
    ax.set(
        title="Per-person missingness by class (full training set)",
        xlabel="Missing predictors per person (%)",
        ylabel="Fraction within each class",
    )
    ax.legend()
    save(fig, "04_row_missingness.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(
        np.log10(np.maximum([r["unique_count"] for r in summaries], 1)),
        bins=22,
        color="#3b6c8d",
    )
    ax.set(
        title="Observed predictor cardinality (full training set)",
        xlabel="log10(distinct nonmissing values); empty columns plotted at 0",
        ylabel="Predictors (count)",
    )
    save(fig, "05_feature_cardinality.png")

    top = ranking[:20][::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        [r["feature_name"] for r in top],
        [r["association_score"] for r in top],
        color="#3b6c8d",
    )
    ax.set(
        title="Largest exploratory univariate associations (full training set)",
        xlabel="max(category eta, |numeric r|, |missingness phi|)",
        ylabel="Predictor",
    )
    save(fig, "06_top_univariate_associations.png")

    numeric_lookup = {r["feature_name"]: i for i, r in enumerate(numeric)}
    selected = [
        numeric_lookup[r["feature_name"]]
        for r in ranking
        if r["feature_name"] in numeric_lookup
    ][:18]
    if not selected:
        selected = list(range(min(18, len(numeric))))
    fig, ax = plt.subplots(figsize=(10, 9))
    if selected:
        image = ax.imshow(
            correlation[np.ix_(selected, selected)], vmin=-1, vmax=1, cmap="RdBu_r"
        )
        labels = [numeric[i]["feature_name"] for i in selected]
        ax.set_xticks(range(len(labels)), labels, rotation=90)
        ax.set_yticks(range(len(labels)), labels)
        fig.colorbar(image, ax=ax, label="Pairwise-complete Pearson r")
    else:
        ax.text(0.5, 0.5, "No eligible numeric predictors", ha="center")
    ax.set(
        title=f"Selected numeric predictors ({len(sample_indices):,} sampled people)",
        xlabel="Predictor",
        ylabel="Predictor",
    )
    save(fig, "07_correlation_heatmap_selected_features.png")

    notable = [r for r in ranking if r["feature_name"] in numeric_lookup][:6]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    summary_lookup = {r["feature_name"]: r for r in summaries}
    for ax, association in zip(axes.flat, notable):
        record = summary_lookup[association["feature_name"]]
        values = data[sample_indices, record["column_index"]]
        observed = np.isfinite(values)
        values, labels = values[observed], y[sample_indices][observed]
        if not len(values):
            continue
        low, high = np.percentile(values, [1, 99])
        if low == high:
            low, high = values.min(), values.max()
        if low == high:
            low, high = low - 0.5, high + 0.5
        bins = np.linspace(low, high, 26)
        for label, color in [(-1, "#3b6c8d"), (1, "#d76e4d")]:
            in_class = labels == label
            if in_class.any():
                ax.hist(
                    values[in_class],
                    bins=bins,
                    weights=np.full(int(in_class.sum()), 1 / in_class.sum()),
                    histtype="step",
                    label=f"{label:+d}",
                    color=color,
                )
        ax.set(
            title=record["feature_name"],
            xlabel="Raw value (sample p01..p99 shown)",
            ylabel="Fraction of observed class",
        )
        ax.legend()
    for ax in list(axes.flat)[len(notable) :]:
        ax.axis("off")
    fig.suptitle(
        "Top numeric associations: sampled distributions; tails outside plot omitted",
        fontsize=12,
    )
    save(fig, "08_selected_numeric_distributions.png")


def write_report(
    output,
    integrity,
    y,
    summaries,
    missing_rows,
    codes,
    categories,
    ranking,
    reviews,
    row_missing,
    duplicates,
    patterns,
    pairs,
    agreements,
    sample_size,
):
    """Write the Markdown EDA report from the computed summaries."""
    n, d = len(y), len(summaries)
    prevalence = float(np.mean(y == 1))
    types = Counter(r["candidate_feature_type"] for r in summaries)
    categorical = [
        r for r in summaries if r["candidate_feature_type"] in CATEGORICAL_TYPES
    ]
    numeric = [r for r in summaries if r["candidate_feature_type"] in NUMERIC_TYPES]
    continuous = [
        r for r in summaries if r["candidate_feature_type"] == "continuous-like numeric"
    ]
    missing_rank = sorted(
        [r for r in missing_rows if r["sufficient_support"]],
        key=lambda r: -r["absolute_difference"],
    )
    strong_missing = [r for r in missing_rows if r["strong_missingness_flag"]]
    constants = [r for r in summaries if r["constant_flag"]]
    near_constants = [r for r in summaries if r["near_constant_flag"]]
    sparse = [r for r in summaries if r["missing_fraction"] > 0.5]
    skewed = [
        r
        for r in numeric
        if np.isfinite(r["upper_lower_tail_ratio"])
        and r["upper_lower_tail_ratio"] >= TAIL_ASYMMETRY_THRESHOLD
        and r["minimum"] >= 0
    ]
    one_hot = sum(r["unique_count"] for r in categorical)
    one_hot_missing = sum(r["missing_count"] > 0 for r in categorical)
    total_width = d - len(categorical) + one_hot + one_hot_missing
    dense_gib = n * total_width * 8 / 1024**3
    stds = [r["std"] for r in numeric if r["std"] > 0]
    ratio = max(stds) / min(stds) if stds else np.nan
    quantiles = np.percentile(row_missing, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    row_stats = [
        {
            "percentile": p,
            "missing_predictors": float(v),
            "missing_fraction": float(v / d),
        }
        for p, v in zip([0, 1, 5, 25, 50, 75, 95, 99, 100], quantiles)
    ]
    code_features = len({r["feature_name"] for r in codes})
    names = (
        lambda rows, count=8: ", ".join(
            "`" + r["feature_name"] + "`" for r in rows[:count]
        )
        or "none"
    )
    sections = [
        "# Exploratory Data Analysis",
        "",
        "This is an observational training-data diagnostic. No data values were recoded,",
        "no final preprocessing was performed, and no classifier was trained.",
        "",
        "## 1. Dataset overview",
        "",
        f'**{n:,} input records, {d} predictors excluding Id, {integrity["y_rows"]:,} target records.**',
        "Inputs: `x_train.csv` and `y_train.csv` only; test/leaderboard data are not used.",
        "",
        "### Reproducibility and definitions",
        "",
        f"- Seed: **{RANDOM_SEED}**. All feature statistics, percentiles, category rates, target summaries,",
        "  exact duplicate checks and missingness statistics use the **full training set**.",
        f"- Correlations use {sample_size:,} rows sampled without replacement with `default_rng`.",
        f"  Categorical agreement uses the first {min(sample_size, CATEGORICAL_SAMPLE_SIZE):,} of that same random sample.",
        "- Blank / NaN / NA cells are missing. Numeric response codes stay observed. Infinity is rejected.",
        "- `column_index` is the original **zero-based CSV position including Id**;",
        "  `feature_index` is zero-based among predictors. Both follow header order.",
        "- `unique_fraction = unique_count / non_missing_count`; standard deviation uses ddof=0.",
        f"- Near constant: >{NEAR_CONSTANT_FRACTION:.0%} of observed values share a value, excluding constants.",
        f"- Supported categories: count >= max({MIN_CATEGORY_SUPPORT}, ceil({MIN_CATEGORY_FRACTION} * observed count)).",
        f"- Missingness comparison: each group >= {MIN_GROUP_SUPPORT} rows; strong flag means an absolute",
        f"  positive-rate difference >= {MISSING_RATE_DIFFERENCE:.0%} (percentage points, not relative percent).",
        f"- Correlation: pairwise-complete Pearson, >= {MIN_PAIR_SUPPORT} co-observed sampled rows;",
        "  no imputation is used. Numeric correlation is only a diagnostic for ordinal candidates.",
        f"- Categorical near duplicate: agreement >= {NEAR_DUPLICATE_AGREEMENT:.0%}, >= {MIN_PAIR_SUPPORT} co-observed",
        "  sampled rows, and both columns vary in that intersection. Dominant categories can inflate agreement.",
        f"- Repeated-digit code candidates: 7/8/9 repeated 1..5 times, observed >= {MIN_CATEGORY_SUPPORT} times.",
        "  This broad heuristic also flags legitimate values; it never assigns a code meaning.",
        f"- Right-tail hypothesis: (p95 - median)/(median - p05) >= {TAIL_ASYMMETRY_THRESHOLD:g},",
        "  with positive denominator and nonnegative minimum. This is not a formal skewness test.",
        "- Full config and software versions: `analysis_metadata.json`. No multiple-testing corrections",
        "  or inferential significance claims are made; all associations are exploratory.",
        "",
        "## 2. Target distribution and class imbalance",
        "",
        f"Negative (-1): **{int((y == -1).sum()):,} ({1 - prevalence:.2%})**. Positive (+1):",
        f"**{int((y == 1).sum()):,} ({prevalence:.2%})**. Majority-only accuracy: **{max(prevalence, 1 - prevalence):.2%}**.",
        "An always-negative classifier has zero positive recall. Raw accuracy therefore hides the minority-class problem.",
        "The mandatory logistic functions expect {0,1}: later map labels with `(y + 1) / 2`.",
        "EDA leaves the original {-1,+1} labels unchanged.",
        "",
        "![Target distribution](figures/01_target_distribution.png)",
        "",
        "## 3. Data integrity",
        "",
        markdown_table(
            [{"check": k, "value": v} for k, v in integrity.items()],
            [("check", "Check"), ("value", "Result")],
            20,
        ),
        "ID uniqueness, missing IDs, identical ID sets, and element-by-element ID order are verified before",
        "target association. A mismatch stops the script; rows are never silently aligned by position.",
        "",
        "## 4. Candidate feature types",
        "",
        markdown_table(
            [{"type": k, "count": v} for k, v in sorted(types.items())],
            [("type", "Candidate type"), ("count", "Features")],
        ),
        "Ordered rules: explicit ID/sequence names (`ID`, `SEQNO`, `_PSU`); no observations;",
        "one observed value; explicit date names (see `DATE_NAMES`); two observed values;",
        f"3..{LOW_CARDINALITY} values -> low-cardinality; {LOW_CARDINALITY + 1}..{MEDIUM_CARDINALITY} -> medium-cardinality;",
        f"integer-like with unique fraction >= {ALMOST_UNIQUE_FRACTION:.0%} -> suspicious high cardinality;",
        f"remaining integer-like <= {DISCRETE_CARDINALITY} values -> ordinal/discrete; remaining numeric -> continuous-like.",
        "These are modeling hypotheses, not semantic labels. High-cardinality floats may still be metadata;",
        "named survey fields receive separate review flags. Known date components remain date-like even with few values.",
        "",
        "## 5. Missing-data analysis",
        "",
        markdown_table(
            [
                {"band": band, "count": count}
                for band, count in missingness_bands(summaries)
            ],
            [("band", "Missingness band"), ("count", "Features")],
        ),
        f"**{len(sparse)} predictors are >50% missing**. Highest missingness:",
        "",
        markdown_table(
            sorted(summaries, key=lambda r: -r["missing_fraction"]),
            [
                ("feature_name", "Feature"),
                ("missing_count", "Missing count"),
                ("missing_fraction", "Missing fraction"),
            ],
        ),
        "Per-person missingness, over all predictors:",
        "",
        markdown_table(
            row_stats,
            [
                ("percentile", "Percentile"),
                ("missing_predictors", "Missing predictors"),
                ("missing_fraction", "Missing fraction"),
            ],
        ),
        f"The p99 row is missing {quantiles[-2]:g} predictors versus a median of {quantiles[4]:g};",
        f"the range is {quantiles[0]:g}..{quantiles[-1]:g}. Missingness varies substantially when these differ.",
        f"Class mean missing fractions: negative {row_missing[y == -1].mean() / d:.2%}, positive {row_missing[y == 1].mean() / d:.2%}.",
        "",
        f"**{len(strong_missing)} features meet the supported missingness-association flag**. Largest supported differences:",
        "",
        markdown_table(
            missing_rank,
            [
                ("feature_name", "Feature"),
                ("missing_count", "Missing n"),
                ("present_count", "Present n"),
                ("positive_rate_when_missing", "Positive rate / missing"),
                ("positive_rate_when_present", "Positive rate / present"),
                ("difference", "Signed difference"),
            ],
        ),
        f"{len(patterns)} groups have identical full-data missingness patterns. See `missingness_patterns.csv`;",
        "shared routing/conditional questions are a hypothesis to verify, not a causal conclusion.",
        "",
        "![Missingness histogram](figures/02_missingness_distribution.png)",
        "![Most missing predictors](figures/03_missingness_by_feature.png)",
        "![Row missingness](figures/04_row_missingness.png)",
        "",
        "## 6. Cardinality and categorical structure",
        "",
        f"Constants: **{len(constants)}** ({names(constants)}). Near constants, separately: **{len(near_constants)}** ({names(near_constants)}).",
        f'Binary: **{types["binary"]}**; candidate categorical including binary: **{len(categorical)}**.',
        f"Almost-unique predictors (unique fraction >= {ALMOST_UNIQUE_FRACTION:.0%}):",
        "",
        markdown_table(
            sorted(
                [
                    r
                    for r in summaries
                    if r["unique_fraction"] >= ALMOST_UNIQUE_FRACTION
                ],
                key=lambda r: -r["unique_fraction"],
            ),
            [
                ("feature_name", "Feature"),
                ("unique_count", "Unique"),
                ("unique_fraction", "Unique / observed"),
                ("candidate_feature_type", "Candidate type"),
            ],
        ),
        "Full categorical frequency and positive-rate tables, including low-support flags: `category_target_summary.csv`.",
        f"Encoding all {len(categorical)} binary/low/medium-cardinality candidates with every observed level",
        f"would create **{one_hot:,} category columns**; explicit missing levels add up to **{one_hot_missing}**.",
        f"Keeping other predictors gives approximately **{total_width:,} total columns** before dropping/review,",
        f"or **{dense_gib:.2f} GiB** for one dense float64 matrix. Copies and optimizer state cost extra.",
        "This is an upper-bound planning estimate for these candidates, not a final encoded matrix:",
        "binary 0/1 mapping, reference-level omission and rare-category merging can reduce width.",
        "Largest categorical candidates:",
        "",
        markdown_table(
            sorted(categorical, key=lambda r: -r["unique_count"]),
            [
                ("feature_name", "Feature"),
                ("unique_count", "Observed levels"),
                ("most_common_fraction", "Modal fraction"),
            ],
        ),
        "![Cardinality](figures/05_feature_cardinality.png)",
        "",
        "## 7. Continuous-feature distributions and scaling",
        "",
        f"Numeric candidates: **{len(numeric)}**, of which **{len(continuous)}** are continuous-like.",
        f"The largest/smallest nonzero numeric standard-deviation ratio is **{ratio:.4g}**.",
        "Largest raw numeric scales (full-data exact percentiles; recorded codes are still included):",
        "",
        markdown_table(
            sorted(numeric, key=lambda r: -r["std"]),
            [
                ("feature_name", "Feature"),
                ("minimum", "Min"),
                ("median", "Median"),
                ("p99", "p99"),
                ("maximum", "Max"),
                ("std", "Std"),
            ],
        ),
        f"**{len(skewed)} nonnegative numeric candidates** meet the right-tail asymmetry heuristic ({names(skewed)}).",
        "Standardize validated numeric measurements using training-fold statistics. Binary encodings do not",
        "require ordinary continuous scaling; numeric category labels must first be encoded or semantically reviewed.",
        "Potential log1p transforms require confirming units and special codes; extreme values are not automatically errors.",
        "",
        "![Selected numeric distributions](figures/08_selected_numeric_distributions.png)",
        "",
        "## 8. Potential encoded missing / refusal / special-response values",
        "",
        f"**{len(codes)} candidate feature/code combinations in {code_features} features.**",
        "Sorted by fraction among observed values. These remain distinct from blank/NaN missingness.",
        "A value such as 7 or 9 can be legitimate; frequency alone does not establish refusal or missingness.",
        "",
        markdown_table(
            sorted(codes, key=lambda r: -r["fraction_of_observed"]),
            [
                ("feature_name", "Feature"),
                ("candidate_code", "Candidate code"),
                ("frequency", "Count"),
                ("fraction_of_observed", "Observed fraction"),
            ],
        ),
        "Check the BRFSS codebook before any recoding. Complete list: `potential_special_codes.csv`.",
        "",
        "## 9. Redundant and correlated predictors",
        "",
        f"Full-data exact duplicate pairs: **{len(duplicates)}**. Sample near-identical categorical pairs: **{len(agreements)}**.",
        markdown_table(
            duplicates, [("feature_a", "Feature A"), ("feature_b", "Feature B")]
        ),
        markdown_table(
            agreements,
            [
                ("feature_a", "Feature A"),
                ("feature_b", "Feature B"),
                ("observed_agreement", "Agreement"),
                ("pair_observed_count", "Co-observed n"),
                ("missingness_agreement", "Mask agreement"),
            ],
        ),
        "High numeric correlations: "
        + "; ".join(
            f'|r| > {t}: {sum(r["absolute_correlation"] > t for r in pairs)} pairs'
            for t in CORRELATION_THRESHOLDS
        )
        + ".",
        "",
        markdown_table(
            pairs,
            [
                ("feature_a", "Feature A"),
                ("feature_b", "Feature B"),
                ("correlation", "Pearson r"),
                ("pair_observed_count", "Co-observed n"),
            ],
        ),
        "Pair lists can reveal overlapping raw/derived representations; correlation is neither identity nor leakage.",
        "Categorical agreement only compares literal codes and may miss equivalent recodings.",
        "Review shared connected pairs before selecting one representative or retaining them with regularization.",
        "",
        "![Numeric correlations](figures/07_correlation_heatmap_selected_features.png)",
        "",
        "## 10. Univariate relationships with MICHD",
        "",
        "Ranking uses max(category eta, absolute numeric point-biserial r, absolute missingness phi).",
        "Category eta is the square root of supported, count-weighted between-category positive-rate variance",
        "divided by supported target variance. Rare categories are excluded from that score, not deleted.",
        "Numeric r uses observed rows; standardized mean difference divides class-mean difference by",
        "sqrt((variance_positive + variance_negative)/2). Missingness phi uses all rows when both groups have support.",
        "Identifiers, date-like, suspicious high-cardinality and named survey metadata are excluded from ranking.",
        "These scores have different biases (including categorical cardinality), are in-sample, and are not",
        "validated predictors or directly comparable model importances. Univariate association is neither",
        "causal importance nor multivariate model importance. Potential leakage flags still require review.",
        "",
        markdown_table(
            ranking,
            [
                ("feature_name", "Feature"),
                ("candidate_feature_type", "Type"),
                ("association_score", "Score"),
                ("score_component", "Largest component"),
                ("positive_observed_count", "Positive n"),
                ("negative_observed_count", "Negative n"),
            ],
            15,
        ),
        "Per-class means/medians, standardized mean differences and all component scores: `target_associations.csv`.",
        "Per-category support and rates: `category_target_summary.csv`. Missingness: `missingness_target.csv`.",
        "",
        "![Univariate ranking](figures/06_top_univariate_associations.png)",
        "",
        "## 11. Features requiring leakage / semantic review",
        "",
        f"**{len(reviews)} predictors** match documented name/provenance/cardinality review rules.",
        "No leakage is asserted from names or associations. Check BRFSS definitions, construction inputs,",
        "survey timing, and whether each field is available at the intended prediction time.",
        "Cardiovascular items may share sources with _MICHD; sampling weights/design fields may encode",
        "survey processing; derived fields may duplicate raw inputs. EDA cannot settle these semantic questions.",
        "",
        markdown_table(
            reviews, [("feature_name", "Feature"), ("reasons", "Reason for review")], 15
        ),
        "Full review list: `semantic_review.csv`. No external codebook was consulted or meanings invented.",
        "",
        "## Recommended preprocessing and cleaning plan",
        "",
        "A. Recommended preprocessing / cleaning actions (proposals for later modeling).",
        "",
    ]
    plans = [
        (
            "IDs and metadata",
            f'{types["identifier"]} ID-like; {sum(r["unique_fraction"] >= ALMOST_UNIQUE_FRACTION for r in summaries)} almost unique',
            "Id plus feature_summary.csv removal flags",
            "Keep Id only as a key; review sequence/date/design fields for removal",
            "High",
        ),
        (
            "Constants / near constants",
            f"{len(constants)} constants; {len(near_constants)} near constants",
            names(constants + near_constants),
            "Drop observed constants only after deciding whether their missingness is useful; validate near-constant removal",
            "High",
        ),
        (
            "Missingness",
            f"{len(sparse)} >50% missing; {len(strong_missing)} supported target differences",
            names(sorted(summaries, key=lambda r: -r["missing_fraction"])),
            "Compare sparse-column removal with categorical missing levels / numeric median imputation and indicators",
            "High",
        ),
        (
            "Possible survey codes",
            f"{code_features} features flagged; no recoding done",
            "potential_special_codes.csv",
            "Verify each code by variable; keep refusal/skip/unknown distinctions when justified",
            "High",
        ),
        (
            "Categorical structure",
            f"{len(categorical)} candidates; up to {one_hot:,} observed-level dummies",
            names(sorted(categorical, key=lambda r: -r["unique_count"])),
            "One-hot nominal levels; map verified binary values; preserve unknown validation levels",
            "High",
        ),
        (
            "Numeric scales",
            f"{len(numeric)} candidates; std ratio {ratio:.4g}",
            names(sorted(numeric, key=lambda r: -r["std"])),
            "Resolve codes and ordinal semantics, then scale genuine numeric variables on training fold only",
            "High",
        ),
        (
            "Redundancy",
            f"{len(duplicates)} exact pairs; {len(pairs)} numeric pairs above {CORRELATION_THRESHOLDS[0]}",
            "exact_duplicate_columns.csv / correlated_pairs.csv",
            "Compare selected raw/derived representatives versus L2 regularization; do not drop merely from r",
            "Medium",
        ),
        (
            "Extreme tails",
            f"{len(skewed)} nonnegative right-tail candidates",
            names(skewed),
            "Test log1p or robust caps only after semantic review, with parameters fit within folds",
            "Medium",
        ),
        (
            "Potential leakage",
            f"{len(reviews)} require provenance/semantic review",
            "semantic_review.csv",
            "Check BRFSS documentation; exclude confirmed target-derived or unavailable-at-prediction fields",
            "High",
        ),
        (
            "Label convention",
            "Stored {-1,+1}; logistic API expects {0,1}",
            "_MICHD",
            "Convert on a modeling copy; retain original labels/IDs for export",
            "High",
        ),
    ]
    sections.append(
        markdown_table(
            [
                dict(zip(("issue", "evidence", "features", "action", "priority"), row))
                for row in plans
            ],
            [
                ("issue", "Issue"),
                ("evidence", "Evidence"),
                ("features", "Affected features / count"),
                ("action", "Proposed action"),
                ("priority", "Priority"),
            ],
            20,
        )
    )
    sections.extend(
        [
            "Fit imputation, category vocabulary, feature selection, transforms and scaling exclusively within",
            "training folds. A high-missingness threshold is a candidate comparison, not an automatic deletion rule.",
            "",
            "## Recommended feature-engineering directions",
            "",
            "B. Candidate feature-engineering directions.",
            "",
            f"- Evidence: {len(strong_missing)} supported missingness contrasts and {len(patterns)} shared-mask groups.",
            "  Hypothesis: individual indicators or a conditional-question-group missing count add signal;",
            "  retain only gains on held-out data.",
            f"- Evidence: {len(categorical)} low/medium categorical candidates. Hypothesis: one-hot encoding avoids",
            "  artificial ordering. Merge rare levels using training-fold support, without conflating them with missing.",
            f"- Evidence: scale ratio {ratio:.4g} and {len(skewed)} nonnegative tail candidates. Hypothesis:",
            "  scaling improves optimization; semantically valid log1p transforms or a few bins can improve fit.",
            f"- Evidence: leading univariate candidates include {names(ranking, 5)}. Hypothesis:",
            "  a small, reviewed subset of numeric quadratic terms or pairwise interactions captures nonlinear effects.",
            "  Marginal association does not establish interactions; predefine a small validation experiment, not all pairs.",
            f"- Evidence: {len(duplicates)} exact duplicate pairs and {len(pairs)} high-r pairs.",
            "  Hypothesis: choosing raw versus derived representatives reduces unstable/redundant dimensions.",
            "",
            "## 14. Recommended validation metrics and protocol",
            "",
            "Use a fixed-seed stratified 80/20 train/validation split implemented with NumPy, then stratified",
            "3-fold comparisons for finalists if runtime permits. Apply the same split to all strategies.",
            "After checking survey metadata, consider an additional group/time split if repeated entities or",
            "collection structure make a random split optimistic. EDA has not established such dependence.",
            "Primary comparison: **balanced accuracy**, with minority-class **F1** and a precision/recall tradeoff.",
            "Report precision (TP/(TP+FP)), recall/sensitivity (TP/(TP+FN)), specificity (TN/(TN+FP)),",
            "F1 (2TP/(2TP+FP+FN)), balanced accuracy ((recall+specificity)/2), and accuracy on every split.",
            "Handle zero denominators explicitly. Secondary: confusion counts and ROC-AUC (tie-aware rank or",
            "threshold integration using NumPy); PR-AUC can complement it under imbalance.",
            "Tune class weights, regularization, encodings and decision thresholds only on training/validation",
            "folds. State the cost/recall objective; a medical setting alone does not define those costs.",
            "Reserve an untouched holdout for the final comparison. The leaderboard is not the primary selection mechanism.",
            "",
            "## Recommended ML strategies",
            "",
            "C. Three recommended ML strategies, ranked for the next experiments.",
            "",
            f"### Strategy 1 — {STRATEGIES[0]}",
            "",
            f"Rank **1**, initial baseline. {n:,} rows favor simple NumPy matrix operations;",
            f"{len(categorical)} categorical candidates, {len(strong_missing)} missingness contrasts and",
            f"{len(pairs)} high-r numeric pairs motivate one-hot encoding, missing indicators and L2 stabilization.",
            "Use reviewed predictors, training-fold numeric imputation/scaling, category vocabularies and",
            "missing indicators. Tune regularization and a validation decision threshold; compare default and",
            "tuned thresholds under imbalance. Advantages: binary likelihood, interpretable coefficients and",
            "compatibility with the mandatory regularized logistic method. Risks: artificial ordering if review",
            "is skipped, collinearity, sparse categories, additive decision surface and low default-threshold recall.",
            f"Cost: moderate; O(NP) per full-batch step, about {dense_gib:.2f} GiB for the conservative encoded",
            "matrix estimate before copies. Chunked gradient accumulation can lower memory in a later custom pipeline.",
            "Hypothesis: a clean, regularized additive classifier supplies a strong, reproducible baseline.",
            "",
            f"### Strategy 2 — {STRATEGIES[1]}",
            "",
            f"Rank **2**. Positive prevalence is {prevalence:.2%}; numeric type/scale/tail evidence suggests",
            "a targeted nonlinear comparison is feasible without a large tree framework. Begin with Strategy 1",
            "representation, add only reviewed numeric transforms/bins or a few selected interactions, and",
            "compare class-weighted loss with unweighted loss plus threshold tuning as separate ablations.",
            "Advantages: explicitly tests minority recall and limited nonlinear effects. Risks: reduced precision,",
            "poorly calibrated weighted probabilities, overfitting feature choices and higher feature memory.",
            "Cost: moderate to high, O(N(P+K)) per step for K added terms; use a small K and NumPy weighted gradients",
            "in a separate future model, without altering mandatory function APIs. Hypothesis: controlled nonlinear",
            "terms and/or cost-sensitive fitting improve the held-out minority tradeoff beyond threshold tuning alone.",
            "",
            f"### Strategy 3 — {STRATEGIES[2]}",
            "",
            f"Rank **3**. The {n:,}-row sample and {len(pairs)} high-r pairs motivate a fast regularized",
            "linear comparison using the mandatory ridge function. Use the same reviewed encoding, imputation",
            "and scaling, retain {-1,+1} regression labels and choose a classification threshold on validation scores.",
            "Advantages: simple NumPy implementation, interpretable coefficients and no iterative convergence tuning",
            "for the direct solver. Risks: squared error is a less natural binary objective, predictions are not",
            "probabilities, normal equations can be ill-conditioned, and positive-class recall still needs monitoring.",
            "Cost: moderate when P is controlled: O(NP^2 + P^3) for direct Gram/solve and O(P^2) Gram storage;",
            "at expanded P this can exceed logistic cost, so benchmark on a fixed subset before a full comparison.",
            "Hypothesis: a simpler regularized linear score captures much of the available signal.",
            "",
            "### Other course methods considered",
            "",
            "Unregularized logistic regression is a useful ablation but less protected against the observed",
            "redundancy and category expansion. Unregularized least squares adds singularity risk with full dummies.",
            "Mandatory MSE-SGD is a low-memory optimization alternative for a linear score, but optimizes squared",
            "error rather than classification likelihood and needs step-size/convergence checks. Mini-batch logistic",
            "optimization can later make Strategies 1/2 cheaper without creating a separate modeling recommendation.",
            "Categorical Naive Bayes could be implemented in NumPy but independence assumptions are questionable",
            "with redundant raw/derived fields. A custom tree/ensemble has higher implementation complexity and",
            "is not prioritized before these controlled linear/feature-transformed comparisons.",
            "",
            "## 16. Conclusions / next steps",
            "",
            "**A. Cleaning:** verify codes/provenance, handle missingness and categories, remove confirmed identity",
            "or constant information, and fit numerical transformations within folds.",
            "**B. Engineering:** validate missing indicators, selected transforms/interactions and redundancy choices.",
            "**C. Modeling:** compare the three ranked strategies on identical stratified validation data,",
            "starting with Strategy 1. These recommendations are hypotheses supported by EDA, not measured model wins.",
            "",
            "Detailed CSVs accompany this report; raw training files and mandatory implementations remain unchanged.",
            "",
        ]
    )
    (output / "eda_report.md").write_text("\n".join(sections), encoding="utf-8")


def main(argv=None):
    """CLI: run the full EDA on the training files and write tables, figures and the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "eda_outputs",
    )
    parser.add_argument("--sample-size", type=int, default=EDA_SAMPLE_SIZE)
    args = parser.parse_args(argv)
    if args.sample_size < 2:
        parser.error("--sample-size must be >= 2")
    print("Loading training CSVs in chunks...", flush=True)
    header, data = load_numeric_csv(args.data_dir / "x_train.csv")
    y_header, targets = load_numeric_csv(args.data_dir / "y_train.csv")
    integrity, y = validate_ids_and_shapes(header, data, y_header, targets)
    print(
        f"IDs verified. Profiling all {len(y):,} rows and {len(header) - 1} predictors...",
        flush=True,
    )
    result = compute_feature_summaries(header, data, y)
    (
        summaries,
        missing_rows,
        codes,
        categories,
        associations,
        reviews,
        row_missing,
        duplicates,
        patterns,
    ) = result
    ranking = sorted(
        [r for r in associations if np.isfinite(r["association_score"])],
        key=lambda r: (-r["association_score"], r["feature_name"]),
    )
    sample_indices = np.random.default_rng(RANDOM_SEED).choice(
        len(y), min(len(y), args.sample_size), replace=False
    )
    print("Computing sampled redundancy diagnostics...", flush=True)
    pairs, agreements, numeric, correlation = analyze_redundancy(
        data, summaries, sample_indices
    )
    output = args.output_dir
    if output.resolve() == args.data_dir.resolve():
        raise ValueError(
            "Use a separate output directory from the input data directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    datasets = [
        ("feature_summary.csv", summaries, ["feature_name"]),
        (
            "missingness_target.csv",
            sorted(
                missing_rows,
                key=lambda r: (
                    not r["sufficient_support"],
                    (
                        -r["absolute_difference"]
                        if np.isfinite(r["absolute_difference"])
                        else 0
                    ),
                ),
            ),
            ["feature_name"],
        ),
        (
            "potential_special_codes.csv",
            codes,
            ["feature_name", "candidate_code", "frequency", "fraction_of_observed"],
        ),
        (
            "category_target_summary.csv",
            categories,
            [
                "feature_name",
                "category",
                "count",
                "positive_count",
                "positive_rate",
                "minimum_support",
                "supported",
            ],
        ),
        (
            "target_associations.csv",
            sorted(
                associations,
                key=lambda r: (
                    not r["ranking_eligible"],
                    (
                        -r["association_score"]
                        if np.isfinite(r["association_score"])
                        else 0
                    ),
                ),
            ),
            ["feature_name"],
        ),
        ("semantic_review.csv", reviews, ["feature_name", "reasons"]),
        (
            "exact_duplicate_columns.csv",
            duplicates,
            ["feature_a", "feature_b", "scope"],
        ),
        (
            "missingness_patterns.csv",
            patterns,
            ["feature_count", "features", "missing_fraction"],
        ),
        (
            "correlated_pairs.csv",
            pairs,
            [
                "feature_a",
                "feature_b",
                "correlation",
                "absolute_correlation",
                "pair_observed_count",
                "sample_rows",
            ],
        ),
        (
            "categorical_agreement.csv",
            agreements,
            [
                "feature_a",
                "feature_b",
                "observed_agreement",
                "pair_observed_count",
                "missingness_agreement",
                "sample_rows",
            ],
        ),
    ]
    for name, records, empty_header in datasets:
        write_csv(output / name, records, empty_header)
    positive_n = int((y == 1).sum())
    (output / "target_summary.txt").write_text(
        f"Rows: {len(y)}\nUnique labels: -1, 1\nNegative count: {len(y) - positive_n}\n"
        f"Positive count: {positive_n}\nNegative proportion: {1 - positive_n / len(y):.10f}\n"
        f"Positive proportion: {positive_n / len(y):.10f}\n"
        f"Majority-class accuracy: {max(positive_n, len(y) - positive_n) / len(y):.10f}\n"
        "Logistic modeling conversion needed: (y + 1) / 2, on a separate modeling copy.\n",
        encoding="utf-8",
    )
    metadata = {
        name: value
        for name, value in globals().items()
        if name.isupper() and isinstance(value, (int, float, tuple))
    }
    metadata.update(
        {
            "input_files": ["x_train.csv", "y_train.csv"],
            "integrity": integrity,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "actual_correlation_sample_rows": len(sample_indices),
            "actual_categorical_sample_rows": min(
                len(sample_indices), CATEGORICAL_SAMPLE_SIZE
            ),
            "requested_sample_size": args.sample_size,
            "full_dataset_statistics": True,
            "sampled_diagnostics_only": [
                "correlation",
                "categorical agreement",
                "distribution figures",
            ],
            "row_missing_percentiles": dict(
                zip(
                    ["min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max"],
                    map(
                        float,
                        np.percentile(row_missing, [0, 1, 5, 25, 50, 75, 95, 99, 100]),
                    ),
                )
            ),
        }
    )
    print("Writing report and figures...", flush=True)
    generate_figures(
        output,
        summaries,
        y,
        row_missing,
        ranking,
        numeric,
        correlation,
        data,
        sample_indices,
    )
    import matplotlib

    metadata["matplotlib"] = matplotlib.__version__
    (output / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    write_report(
        output,
        integrity,
        y,
        summaries,
        missing_rows,
        codes,
        categories,
        ranking,
        reviews,
        row_missing,
        duplicates,
        patterns,
        pairs,
        agreements,
        len(sample_indices),
    )
    types = Counter(r["candidate_feature_type"] for r in summaries)
    print(
        f"\nEDA complete.\nRows: {len(y):,}\nFeatures: {len(summaries)}\nPositive prevalence: {positive_n / len(y):.2%}\n"
        f'Constant features: {sum(r["constant_flag"] for r in summaries)}\n'
        f'>50% missing features: {sum(r["missing_fraction"] > 0.5 for r in summaries)}\n'
        f"Candidate categorical features: {sum(types[k] for k in CATEGORICAL_TYPES)}\n"
        f'Candidate continuous features: {types["continuous-like numeric"]}\n'
        f'Candidate ID-like features: {types["identifier"] + types["high-cardinality suspicious"]}\n'
        f'Features requiring semantic/code review: {len({r["feature_name"] for r in reviews + codes})}'
    )
    for rank, strategy in enumerate(STRATEGIES, 1):
        print(f"Recommended Strategy {rank}: {strategy}")
    print(f'Full report: {output / "eda_report.md"}')


if __name__ == "__main__":
    main()
