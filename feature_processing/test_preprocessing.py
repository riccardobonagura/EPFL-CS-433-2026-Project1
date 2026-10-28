"""Training isolation, schema safety, transparent state and OOF encoding checks."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from feature_processing import (
    fit_preprocessor,
    fit_transform,
    get_config,
    load_state,
    save_state,
    transform,
    transform_with_diagnostics,
)
from feature_processing.target_encoding import (
    cross_fit_target_encode,
    fit_target_encoder,
    transform_target_encoder,
)


class PreprocessingTests(unittest.TestCase):
    """Checks of the preprocessing package on small synthetic matrices."""

    def config(self, **options):
        """Return a test config with the given overrides."""
        settings = dict(
            rare_min_count=1,
            rare_min_fraction=0,
            canonical=False,
            semantic_types={"num": "numeric", "cat": "categorical"},
        )
        settings.update(options)
        return get_config(settings.pop("profile", "curated_v1"), **settings)

    def test_training_only_medians_scaling_unknown_categories_and_repeatability(self):
        """Statistics use training rows; unseen levels go to "other"; reruns match."""
        train = np.array([[0, 1], [2, 2], [np.nan, 1]], dtype=float)
        state = fit_preprocessor(train, ["num", "cat"], config=self.config())
        validation = np.array([[1000, 99], [np.nan, np.nan]])
        before = json.dumps(state.__dict__, sort_keys=True)
        result, diagnostics = transform_with_diagnostics(
            validation, ["num", "cat"], state
        )
        self.assertEqual(state.medians["num"], 1)
        self.assertEqual(state.means["num"], 1)
        self.assertAlmostEqual(state.stds["num"], np.std([0, 2, 1]))
        self.assertAlmostEqual(
            float(result[0, state.output_features.index("num__scaled")]),
            999 / np.std([0, 2, 1]),
            places=3,
        )
        self.assertEqual(result[1, state.output_features.index("num__scaled")], 0)
        self.assertEqual(result[0, state.output_features.index("cat____OTHER__")], 1)
        self.assertEqual(result[0, state.output_features.index("cat____MISSING__")], 0)
        self.assertEqual(result[1, state.output_features.index("cat____MISSING__")], 1)
        self.assertEqual(diagnostics["unknown_category_total"], 1)
        np.testing.assert_array_equal(
            result, transform(validation, ["num", "cat"], state)
        )
        self.assertEqual(before, json.dumps(state.__dict__, sort_keys=True))
        self.assertEqual(result.dtype, np.float32)

    def test_missing_and_rare_categories_are_distinct_and_reference_is_stored(self):
        """Missing and rare levels get separate columns; the reference is stored."""
        train = np.array([1, 1, 1, 1, 1, 2, 2, np.nan])[:, None]
        result, state = fit_transform(
            train, ["cat"], config=self.config(rare_min_count=3)
        )
        self.assertEqual(state.specs["cat"]["reference"], "1")
        self.assertEqual(state.rare_category_maps["cat"], ["2"])
        self.assertEqual(result[5, state.output_features.index("cat____OTHER__")], 1)
        self.assertEqual(result[5, state.output_features.index("cat____MISSING__")], 0)
        self.assertEqual(result[7, state.output_features.index("cat____MISSING__")], 1)

    def test_ids_excluded_and_columns_can_be_reordered(self):
        """Identifiers are dropped; transform accepts reordered columns."""
        x = np.array([[10, 1, 8], [11, 2, 9], [12, 3, 10]], dtype=float)
        names = ["Id", "num", "SEQNO"]
        result, state = fit_transform(x, names, config=self.config())
        self.assertIn("Id", state.dropped_features)
        self.assertIn("SEQNO", state.dropped_features)
        self.assertFalse(
            any(name.startswith(("Id__", "SEQNO__")) for name in state.output_features)
        )
        reordered = transform(x[:, [2, 0, 1]], ["SEQNO", "Id", "num"], state)
        np.testing.assert_array_equal(result, reordered)
        self.assertNotIn("Id", state.missing_count_sources)

    def test_constants_and_duplicate_presence_indicators(self):
        """Constant fields are dropped; duplicate indicators are not repeated."""
        x = np.array([[1, 2, 1], [np.nan, np.nan, 2], [1, 2, 3], [np.nan, np.nan, 4]])
        result, state = fit_transform(
            x, ["constant_a", "constant_b", "num"], config=self.config()
        )
        self.assertEqual(state.specs["constant_a"]["drop_reason"], "constant")
        self.assertEqual(state.specs["constant_b"]["drop_reason"], "constant")
        present = [name for name in state.output_features if name.endswith("__present")]
        self.assertEqual(present, ["constant_a__present"])
        np.testing.assert_array_equal(
            result[:, state.output_features.index(present[0])], [1, 0, 1, 0]
        )

    def test_configured_exact_duplicates_and_high_correlations_are_distinct(self):
        """Only exact declared duplicates are removed, not merely correlated fields."""
        x = np.column_stack(
            [np.arange(8), np.arange(8), np.arange(8) + np.arange(8) ** 2 * 0.001]
        )
        config = self.config(
            semantic_types={"num": "numeric", "alias": "numeric", "near": "numeric"},
            exact_duplicates={"alias": "num"},
        )
        _, state = fit_transform(x, ["num", "alias", "near"], config=config)
        self.assertEqual(state.specs["alias"]["drop_reason"], "exact_duplicate")
        self.assertIn("near__scaled", state.output_features)
        self.assertNotIn("alias__present", state.output_features)

    def test_sparse_raw_value_removal_preserves_presence(self):
        """Very sparse fields are dropped but keep a presence indicator."""
        sparse = np.full(100, np.nan)
        sparse[:2] = [1, 2]
        result, state = fit_transform(sparse[:, None], ["sparse"], config=self.config())
        self.assertEqual(state.specs["sparse"]["drop_reason"], "sparse_raw_value")
        self.assertIn("sparse__present", state.output_features)
        self.assertEqual(
            result[:, state.output_features.index("sparse__present")].sum(), 2
        )

    def test_quarantine_also_excludes_presence_aggregates_and_engineering(self):
        """Quarantined fields feed no indicator, missing count or engineered feature."""
        x = np.array([[0, 1], [1, np.nan], [2, np.nan]], dtype=float)
        config = self.config(engineer=True, squares={"blocked_squared": "HAREHAB1"})
        result, state = fit_transform(x, ["num", "HAREHAB1"], config=config)
        self.assertEqual(state.specs["HAREHAB1"]["drop_reason"], "quarantine")
        self.assertNotIn("HAREHAB1", state.missing_count_sources)
        changed = x.copy()
        changed[:, 1] = [np.nan, 999, 888]
        np.testing.assert_array_equal(
            result, transform(changed, ["num", "HAREHAB1"], state)
        )
        self.assertNotIn("blocked_squared", state.output_features)
        _, enabled = fit_transform(
            x, ["num", "HAREHAB1"], config=self.config(include_quarantined=True)
        )
        self.assertNotEqual(enabled.specs["HAREHAB1"].get("drop_reason"), "quarantine")

    def test_codes_only_change_under_explicit_verified_mapping(self):
        """Survey codes change only through an explicit verified mapping."""
        x = np.array([[1.0], [2], [88], [99]])
        _, raw = fit_transform(x, ["num"], config=self.config(special_code_map={}))
        self.assertEqual(raw.medians["num"], 45)
        _, decoded = fit_transform(
            x,
            ["num"],
            config=self.config(special_code_map={"num": {88: "missing", 99: 3}}),
        )
        self.assertEqual(decoded.medians["num"], 2)
        categorical, state = fit_transform(
            x, ["num"], config=self.config(special_code_map={"num": {99: "category"}})
        )
        self.assertEqual(state.specs["num"]["kind"], "categorical")
        self.assertTrue(np.isfinite(categorical).all())

    def test_verified_binary_mapping_keeps_missing_and_unknown_distinct(self):
        """Binary maps keep missing and unknown codes as separate columns."""
        x = np.array([[1], [2], [7], [9], [1]], dtype=float)
        _, state = fit_transform(x, ["BPMEDS"], config=self.config())
        result = transform(np.array([[1], [2], [np.nan], [99]]), ["BPMEDS"], state)
        np.testing.assert_array_equal(
            result[:, state.output_features.index("BPMEDS__binary")], [1, 0, 0, 0]
        )
        np.testing.assert_array_equal(
            result[:, state.output_features.index("BPMEDS__missing")], [0, 0, 1, 0]
        )
        np.testing.assert_array_equal(
            result[:, state.output_features.index("BPMEDS____OTHER__")], [0, 0, 0, 1]
        )

    def test_medium_cardinality_not_silently_numeric(self):
        """Medium-cardinality codes are not treated as numeric by default."""
        _, state = fit_transform(np.arange(10)[:, None], ["cat"], config=self.config())
        self.assertEqual(state.specs["cat"]["drop_reason"], "unused_medium_cardinality")
        self.assertNotIn("cat__scaled", state.output_features)

    def test_unverified_high_cardinality_numeric_requires_opt_in(self):
        """Unverified numeric fields are used only when explicitly allowed."""
        _, state = fit_transform(
            (np.arange(100) / 10)[:, None], ["unknown"], config=self.config()
        )
        self.assertEqual(state.specs["unknown"]["drop_reason"], "unverified_numeric")

    def test_numeric_log_clipping_and_statistics_are_fit_on_training(self):
        """Clip limits, log1p and scaling are fitted on training rows."""
        x = np.array([[0], [1], [100]], dtype=float)
        config = self.config(log1p_features=["num"], clip_features={"num": (0, 50)})
        _, state = fit_transform(x, ["num"], config=config)
        self.assertEqual(state.specs["num"]["numeric"]["clip"], [0, 1])
        actual = transform(np.array([[10000.0]]), ["num"], state)
        expected = transform(np.array([[1.0]]), ["num"], state)
        np.testing.assert_array_equal(actual, expected)
        with self.assertRaises(ValueError):
            fit_transform(np.array([[-1.0], [1]]), ["num"], config=config)

    def test_state_roundtrip_preserves_order_dtype_and_values(self):
        """A saved and reloaded state reproduces the same matrix."""
        x = np.array([[0, 1], [2, 2], [np.nan, 1]])
        result, state = fit_transform(
            x, ["num", "cat"], config=self.config(output_dtype=np.float64)
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.json"
            save_state(state, path)
            restored = load_state(path)
            np.testing.assert_array_equal(
                result, transform(x, ["num", "cat"], restored)
            )
        self.assertEqual(state.output_features, restored.output_features)
        self.assertEqual(len(set(state.output_features)), result.shape[1])
        self.assertEqual(result.dtype, np.float64)

    def test_engineering_is_small_explicit_and_requires_validated_sources(self):
        """Engineered features are only the configured ones, from valid sources."""
        x = np.array([[20, 1], [30, 2], [40, 1], [50, 2]], dtype=float)
        config = self.config(
            semantic_types={"_AGE80": "numeric", "cat": "categorical"},
            engineer=True,
            squares={"age_squared": "_AGE80"},
            interactions={"blocked": ["_AGE80", "cat"]},
            dummy_interactions={"age_x_cat2": ["_AGE80__scaled", "cat__2"]},
        )
        result, state = fit_transform(x, ["_AGE80", "cat"], config=config)
        self.assertIn("age_squared", state.output_features)
        self.assertIn("age_x_cat2", state.output_features)
        self.assertNotIn("blocked", state.output_features)
        self.assertEqual(len(state.engineered_feature_config["recipes"]), 2)
        self.assertTrue(np.isfinite(result).all())

    def test_condition_count_requires_explicit_verified_binary_sources(self):
        """The condition count needs verified binary sources."""
        x = np.array([[1, 2], [2, 1], [1, 1], [2, 2]], dtype=float)
        config = self.config(
            engineer=True,
            binary_maps={"a": {1: 1, 2: 0}, "b": {1: 1, 2: 0}},
            major_condition_features=["a", "b"],
            squares={},
        )
        _, state = fit_transform(x, ["a", "b"], config=config)
        self.assertIn("major_condition_count", state.output_features)
        config["major_condition_features"] = ["unknown"]
        with self.assertRaises(ValueError):
            fit_transform(x, ["a", "b"], config=config)

    def test_age_profiles_keep_only_selected_representation(self):
        """Only the chosen age representation is kept."""
        x = np.column_stack([np.arange(18, 38), np.tile([1, 2, 3, 4], 5)])
        config = get_config("curated_v1", rare_min_count=1, rare_min_fraction=0)
        _, state = fit_transform(x, ["_AGE80", "_AGEG5YR"], config=config)
        self.assertIn("_AGE80__scaled", state.output_features)
        self.assertEqual(
            state.specs["_AGEG5YR"]["drop_reason"], "redundant_representation"
        )
        config["group_choices"]["age"] = "age_group_onehot"
        _, state = fit_transform(x, ["_AGE80", "_AGEG5YR"], config=config)
        self.assertNotIn("_AGE80__scaled", state.output_features)
        self.assertIn("_AGEG5YR", state.categorical_features)

    def test_memory_warning_occurs_before_dense_allocation(self):
        """A large output warns before the matrix is allocated."""
        x = np.array([[1], [2], [3]], dtype=float)
        state = fit_preprocessor(
            x, ["num"], config=self.config(memory_warning_gib=1e-12)
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            transform(x, ["num"], state)
        self.assertTrue(
            any("Dense processed allocation" in str(item.message) for item in caught)
        )

    def test_target_encoding_uses_fold_prior_and_excludes_own_target(self):
        """Out-of-fold encoding never uses a row's own label."""
        x = np.arange(15, dtype=float)[:, None]
        y = np.tile([0, 1, 0], 5)
        first, encoder = cross_fit_target_encode(
            x,
            y,
            ["cat"],
            n_folds=3,
            seed=42,
            alpha=20,
            rare_min_count=1,
            rare_min_fraction=0,
        )
        for row in (0, 6, 14):
            changed = y.copy()
            changed[row] = 1 - changed[row]
            other, _ = cross_fit_target_encode(
                x,
                changed,
                ["cat"],
                n_folds=3,
                seed=42,
                alpha=20,
                rare_min_count=1,
                rare_min_fraction=0,
            )
            self.assertEqual(first[row, 0], other[row, 0])
        unseen = transform_target_encoder(np.array([[999.0]]), encoder, ["cat"])
        self.assertEqual(unseen[0, 0], y.mean())

    def test_target_encoding_smoothing_formula(self):
        """Encoded values follow (n * rate + alpha * prior) / (n + alpha)."""
        state = fit_target_encoder(
            np.array([1, 1, 2, 2]),
            np.array([1, 1, 0, 0]),
            ["cat"],
            alpha=2,
            rare_min_count=1,
            rare_min_fraction=0,
        )
        result = transform_target_encoder(np.array([1, 2]), state, ["cat"])
        np.testing.assert_allclose(result[:, 0], [0.75, 0.25])

    def test_compact_profile_is_oof_and_reloadable_for_validation(self):
        """The target-encoded profile is out-of-fold and reloads for validation."""
        x = np.arange(18, dtype=float)[:, None]
        y = np.tile([0, 1], 9)
        config = self.config(
            profile="compact_target_encoded", max_onehot_levels=2, target_folds=3
        )
        training, state = fit_transform(x, ["cat"], y=y, config=config)
        self.assertEqual(state.target_features, ["cat"])
        changed_y = y.copy()
        changed_y[0] = 1 - changed_y[0]
        other, _ = fit_transform(x, ["cat"], y=changed_y, config=config)
        self.assertEqual(
            training[0, state.output_features.index("cat__target")],
            other[0, state.output_features.index("cat__target")],
        )
        with self.assertRaises(ValueError):
            transform(x, ["cat"], state)
        validation = np.array([[99.0], [np.nan]])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.json"
            save_state(state, path)
            np.testing.assert_array_equal(
                transform(validation, ["cat"], state),
                transform(validation, ["cat"], load_state(path)),
            )

    def test_input_validation_and_no_mutation(self):
        """Invalid inputs raise ValueError; inputs are not modified."""
        x = np.array([[0, 1], [2, 2], [np.nan, 1]])
        original = x.copy()
        fit_transform(x, ["num", "cat"], config=self.config())
        np.testing.assert_array_equal(x, original)
        for values, names in [
            (x[:, 0], ["num"]),
            (x, ["num"]),
            (x, ["a", "a"]),
            (np.array([[np.inf]]), ["num"]),
        ]:
            with self.assertRaises(ValueError):
                fit_preprocessor(values, names)
        state = fit_preprocessor(x, ["num", "cat"], config=self.config())
        with self.assertRaises(ValueError):
            transform(x, ["wrong", "cat"], state)

    def test_runners_save_reusable_outputs_and_preserve_unlabeled_id_order(self):
        """Runner outputs reload correctly and keep the test ID order."""
        from feature_processing.run_preprocessing import main as run
        from feature_processing.evaluate_processing import main as evaluate

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data, output = root / "data", root / "outputs"
            data.mkdir()
            n = 240
            age = 20 + np.arange(n) % 60
            raw = np.column_stack(
                [
                    np.arange(n),
                    age,
                    1 + np.arange(n) % 5,
                    20 + np.arange(n) % 20,
                    1 + np.arange(n) % 2,
                ]
            )
            names = ["Id", "_AGE80", "GENHLTH", "_BMI5", "BPMEDS"]
            np.savetxt(
                data / "x_train.csv",
                raw,
                delimiter=",",
                header=",".join(names),
                comments="",
            )
            np.savetxt(
                data / "y_train.csv",
                np.column_stack([np.arange(n), np.where(age > 50, 1, -1)]),
                delimiter=",",
                header="Id,_MICHD",
                comments="",
            )
            with (data / "x_test.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.writer(stream)
                writer.writerow(names)
                writer.writerows([["second", 30, 2, 25, 1], ["first", 60, 4, 31, 2]])
            original = {path.name: path.read_bytes() for path in data.iterdir()}
            with contextlib.redirect_stdout(io.StringIO()):
                run(
                    [
                        "--data-dir",
                        str(data),
                        "--output-dir",
                        str(output),
                        "--save-arrays",
                        "--test-csv",
                        str(data / "x_test.csv"),
                    ]
                )
                evaluate(
                    [
                        "--data-dir",
                        str(data),
                        "--output-dir",
                        str(output),
                        "--max-rows",
                        "240",
                        "--max-updates",
                        "2",
                    ]
                )
            np.testing.assert_array_equal(
                np.load(output / "ids_test.npy"), ["second", "first"]
            )
            self.assertEqual(np.load(output / "X_test_processed.npz")["X"].shape[0], 2)
            self.assertTrue((output / "preprocessing_state.json").is_file())
            self.assertTrue((output / "processing_report.md").is_file())
            with (output / "ablation_results.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 7)
            self.assertTrue(all("default_balanced_accuracy" in row for row in rows))
            self.assertTrue(
                all(path.read_bytes() == original[path.name] for path in data.iterdir())
            )


if __name__ == "__main__":
    unittest.main()
