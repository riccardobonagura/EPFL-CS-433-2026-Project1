"""Focused checks of EDA integrity rules and statistical calculations."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

import eda


class EDATests(unittest.TestCase):
    """Unit and end-to-end checks of eda.py on small synthetic CSVs."""

    def test_chunk_loader_missing_values_and_reordered_columns(self):
        """Chunked loading handles missing cells and reordered columns."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "x.csv"
            path.write_text(
                "value,Id,other\n1,10,\n2,11,NaN\n3,12,NA\n", encoding="utf-8"
            )
            header, data = eda.load_numeric_csv(path)
            self.assertEqual(header, ["value", "Id", "other"])
            np.testing.assert_array_equal(data[:, 1], [10, 11, 12])
            self.assertTrue(np.isnan(data[:, 2]).all())
            self.assertTrue(data.flags.f_contiguous)

    def test_loader_rejects_nonnumeric_and_malformed_records(self):
        """Loading rejects non-numeric values and rows of the wrong width."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "x.csv"
            for content in [
                "Id,value\n0,oops\n",
                "Id,value\n0,1,2\n",
                "Id,value\n0,inf\n",
            ]:
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content), self.assertRaises(ValueError):
                    eda.load_numeric_csv(path)

    def test_id_validation_reordered_columns(self):
        """ID checks pass when the Id column is not first."""
        x = np.array([[4, 10], [5, 11]], dtype=float)
        target = np.array([[-1, 10], [1, 11]], dtype=float)
        integrity, y = eda.validate_ids_and_shapes(
            ["feature", "Id"], x, ["_MICHD", "Id"], target
        )
        self.assertTrue(integrity["exact_id_order"])
        np.testing.assert_array_equal(y, [-1, 1])

    def test_id_validation_rejects_duplicate_missing_unmatched_or_reordered_ids(self):
        """ID checks reject duplicate, missing, unmatched or reordered IDs."""
        target = np.array([[10, -1], [11, 1]], dtype=float)
        for ids in [[10, 10], [10, np.nan], [10, 12], [11, 10]]:
            x = np.column_stack([ids, [3, 4]])
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                eda.validate_ids_and_shapes(["Id", "a"], x, ["Id", "_MICHD"], target)

    def test_target_validation(self):
        """Targets must be finite, in {-1, +1}, with both classes present."""
        x = np.array([[0, 4], [1, 5]], dtype=float)
        for labels in [[0, 1], [-1, np.nan], [-1, -1]]:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                eda.validate_ids_and_shapes(
                    ["Id", "a"], x, ["Id", "_MICHD"], np.column_stack([[0, 1], labels])
                )

    def test_candidate_type_rules(self):
        """Fields are classified (binary, categorical, numeric, ...) by the documented rules."""
        examples = [
            ("SEQNO", 300, 1.0, True, 300, "identifier"),
            ("empty", 0, np.nan, False, 0, "unknown"),
            ("constant", 1, 0.01, True, 100, "constant"),
            ("IMONTH", 12, 0.04, True, 300, "date-like"),
            ("flag", 2, 0.01, True, 300, "binary"),
            ("codes", 4, 0.02, True, 300, "low-cardinality categorical"),
            ("codes", 20, 0.1, True, 300, "medium-cardinality categorical"),
            ("suspicious", 300, 1.0, True, 300, "high-cardinality suspicious"),
            ("discrete", 80, 0.1, True, 800, "ordinal-like / discrete numeric"),
            ("measurement", 300, 1.0, False, 300, "continuous-like numeric"),
        ]
        for *arguments, expected in examples:
            with self.subTest(expected=expected):
                self.assertEqual(eda.infer_candidate_type(*arguments)[0], expected)

    def test_full_feature_statistics_categories_missingness_and_duplicates(self):
        """Per-field statistics, categories, missingness and duplicates are correct."""
        n = 300
        labels = np.tile([-1, 1], n // 2)
        binary = np.tile([1.0, 9.0], n // 2)
        conditional = np.where(labels == 1, np.nan, 2.0)
        near_constant = np.ones(n)
        near_constant[-1] = 2
        data = np.column_stack(
            [
                np.arange(n),
                binary,
                binary,
                conditional,
                np.full(n, np.nan),
                near_constant,
            ]
        )
        original = data.copy()
        result = eda.compute_feature_summaries(
            ["Id", "codes", "duplicate", "conditional", "empty", "near"], data, labels
        )
        (
            summaries,
            missing_rows,
            codes,
            categories,
            associations,
            _,
            row_missing,
            duplicates,
            _,
        ) = result
        lookup = {r["feature_name"]: r for r in summaries}
        self.assertEqual(len(summaries), 5)
        self.assertEqual(lookup["codes"]["unique_count"], 2)
        self.assertEqual(lookup["codes"]["most_common_frequency"], 150)
        self.assertEqual(lookup["codes"]["median"], 5)
        self.assertTrue(lookup["near"]["near_constant_flag"])
        self.assertFalse(lookup["conditional"]["near_constant_flag"])
        self.assertTrue(lookup["conditional"]["constant_flag"])
        self.assertEqual(lookup["empty"]["candidate_feature_type"], "unknown")
        self.assertEqual(lookup["conditional"]["missing_fraction"], 0.5)
        missing = next(r for r in missing_rows if r["feature_name"] == "conditional")
        self.assertEqual(missing["positive_rate_when_missing"], 1)
        self.assertEqual(missing["positive_rate_when_present"], 0)
        self.assertTrue(missing["strong_missingness_flag"])
        category = next(
            r for r in categories if r["feature_name"] == "codes" and r["category"] == 9
        )
        self.assertEqual(category["positive_count"], 150)
        self.assertEqual(category["positive_rate"], 1)
        self.assertTrue(category["supported"])
        self.assertEqual(
            next(r for r in codes if r["feature_name"] == "codes")["candidate_code"], 9
        )
        self.assertEqual(
            next(r for r in associations if r["feature_name"] == "codes")[
                "categorical_eta"
            ],
            1,
        )
        self.assertTrue(
            any(
                r["feature_a"] == "codes" and r["feature_b"] == "duplicate"
                for r in duplicates
            )
        )
        np.testing.assert_array_equal(row_missing, 1 + (labels == 1))
        np.testing.assert_array_equal(data, original)

    def test_pairwise_complete_correlations(self):
        """Correlations use only rows where both fields are present."""
        data = np.random.default_rng(433).normal(size=(40, 4))
        data[::3, 0] = np.nan
        data[::4, 1] = np.nan
        data[:, 3] = 1
        correlation, support = eda.pairwise_complete_correlation(data, min_support=2)
        for i in range(3):
            for j in range(3):
                observed = np.isfinite(data[:, i]) & np.isfinite(data[:, j])
                expected = np.corrcoef(data[observed, i], data[observed, j])[0, 1]
                self.assertAlmostEqual(correlation[i, j], expected)
                self.assertEqual(support[i, j], observed.sum())
        self.assertTrue(np.isnan(correlation[3]).all())
        correlation, _ = eda.pairwise_complete_correlation(data, min_support=100)
        self.assertTrue(np.isnan(correlation).all())

    def test_missingness_bands_cover_boundaries_once(self):
        """Every field falls in exactly one missingness band, boundaries included."""
        summaries = [
            {"missing_fraction": value}
            for value in [0, 0.01, 0.1, 0.3, 0.5, 0.8, 0.95, 1]
        ]
        self.assertEqual(
            [count for _, count in eda.missingness_bands(summaries)], [1] * 8
        )

    def test_end_to_end_outputs_are_deterministic(self):
        """Two full EDA runs produce identical outputs."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            n = 600
            ids = np.arange(n)
            labels = np.tile([-1, 1], n // 2)
            measurement = np.random.default_rng(433).normal(size=n)
            data = np.column_stack(
                [
                    ids,
                    measurement,
                    measurement,
                    np.tile([1, 2], n // 2),
                    np.where(labels == 1, np.nan, 3),
                    np.full(n, np.nan),
                ]
            )
            np.savetxt(
                root / "x_train.csv",
                data,
                delimiter=",",
                header="Id,measurement,duplicate,flag,conditional,empty",
                comments="",
            )
            np.savetxt(
                root / "y_train.csv",
                np.column_stack([ids, labels]),
                delimiter=",",
                header="Id,_MICHD",
                comments="",
            )
            for name in ["first", "second"]:
                with contextlib.redirect_stdout(io.StringIO()):
                    eda.main(
                        [
                            "--data-dir",
                            str(root),
                            "--output-dir",
                            str(root / name),
                            "--sample-size",
                            "600",
                        ]
                    )
            first, second = root / "first", root / "second"
            files = sorted(
                path.relative_to(first) for path in first.rglob("*") if path.is_file()
            )
            self.assertEqual(sum(path.suffix == ".png" for path in files), 8)
            for path in files:
                with self.subTest(path=str(path)):
                    self.assertEqual(
                        (first / path).read_bytes(), (second / path).read_bytes()
                    )
            report = (first / "eda_report.md").read_text(encoding="utf-8")
            self.assertEqual(report.count("### Strategy "), 3)
            self.assertIn("## Recommended preprocessing and cleaning plan", report)
            self.assertIn("## Recommended ML strategies", report)


if __name__ == "__main__":
    unittest.main()
