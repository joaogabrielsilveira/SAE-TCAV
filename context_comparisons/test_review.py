"""Regression checks for the independently reviewed comparison failures."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from context_comparisons import native, provenance, reporting, study
from context_comparisons.diagnostics import factor_summary, reconstruction_summary
from context_comparisons.geometry import retrieval_diagnostics


class ReviewTests(unittest.TestCase):
    def test_new_predictions_do_not_claim_a_validation_tuned_threshold(self):
        reference = {"build": "last_3", "reference": 2010,
                     "population": {"patient_id": np.array(["a"]), "y": np.array([0])}}
        with patch.object(study.data, "evaluation_rows", return_value=np.array([0])):
            result = study.frame(reference, 2011, np.array([[.2]] * 3), "native", .1, .5)
        self.assertTrue(result["tuned_threshold"].isna().all())

    def test_native_order_restores_values_and_rejects_invalid_mappings(self):
        states = np.arange(4 * 192).reshape(4, 192).astype(float)
        classifier = SimpleNamespace(ensemble_configurations_=[
            SimpleNamespace(subsample_samples_configuration=np.array([2, 0, 3, 1]))])
        ordered = native.ordered_states({"classifier": classifier}, [states], 4)
        np.testing.assert_array_equal(ordered, states[[1, 3, 0, 2]])
        for mapping in ([0, 0, 2, 3], [0, 1, 2], [0, 1, 2, 4]):
            classifier.ensemble_configurations_[0].subsample_samples_configuration = mapping
            with self.assertRaises(ValueError):
                native.ordered_states({"classifier": classifier}, [states], 4)

    def test_cosine_guard_detects_radial_and_within_stratum_collapse(self):
        years, labels = np.array([2009] * 4), np.array([0, 0, 1, 1])
        patients, outcomes, probabilities = np.eye(2), np.array([0, 1]), np.array([.1, .9])
        for vectors in (np.array([[1., 0.], [2., 0.], [3., 0.], [4., 0.]]),
                        np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]]), np.zeros((4, 2))):
            result = retrieval_diagnostics(vectors, patients, years, labels, outcomes, probabilities, .5, size=2)
            self.assertFalse(result["any_discriminating_stratum"])
        result = retrieval_diagnostics(np.array([[1., 0.], [0., 1.], [1., 0.], [0., 1.]]),
                                       patients, years, labels, outcomes, probabilities, .5, size=2)
        self.assertTrue(result["any_discriminating_stratum"])

    def test_actual_scores_can_tie_even_with_distinct_context_directions(self):
        result = retrieval_diagnostics(np.array([[1., 1.], [1., -1.], [1., 1.], [1., -1.]]),
            np.array([[1., 0.]]), np.array([2009] * 4), np.array([0, 0, 1, 1]),
            np.array([0]), np.array([.2]), .5, size=2)
        self.assertFalse(result["any_discriminating_stratum"])
        self.assertTrue(all(record["boundary_tied_rows"] == 2 for record in result["strata"]))

    def test_native_cosine_collapse_uses_full_context_without_measuring_exclusions(self):
        reference = {"build": "last_3", "reference": 2010, "context_rows": np.arange(4),
                     "population": {"year": np.array([2010] * 8 + [2011]),
                                    "patient_id": np.array([str(i) for i in range(9)]),
                                    "y": np.array([0, 0, 1, 1, 0, 1, 0, 1, 0])}}
        def rows(reference, year, part):
            return np.array([4, 5]) if part == "development" else np.array([6, 7])
        with patch.object(study.original_study, "reference_info", return_value=({"death_threshold": {"last_3": .5}}, [])), \
             patch.object(study.measures, "load_build", return_value={"reference_mortality": .2}), \
             patch.object(study.paired, "development_rows", side_effect=rows), \
             patch.object(study.data, "evaluation_rows", return_value=np.array([8])), \
             patch.object(study.data, "cached_predictor_rows", return_value=(np.array([.2, .8]), None)), \
             patch.object(study.paired, "predictions", return_value=np.array([[.2]] * 3)), \
             patch.object(study.paired, "neighborhood_priorities") as effects:
            frames, result, priorities = study.evaluate_variant(reference, 2010, "concepts",
                (np.array([[1., 0.], [2., 0.], [3., 0.], [4., 0.]]), np.eye(2)),
                "cumulative", "native_context", {"provenance": {"sample_size": 2, "draws": 1}})
        effects.assert_not_called()
        self.assertTrue(result["context_representation_varies"])
        self.assertIn("cosine", result["fallback_reason"])
        for frame in frames:
            self.assertTrue(frame["fallback"].all())
            self.assertFalse(frame["common_size_eligible"].any())
            self.assertTrue((frame["context_size"] == 4).all())

    def test_waiting_entry_point_does_not_preload_experiment_modules(self):
        code = ("import importlib, sys; importlib.import_module('context_comparisons.__main__'); "
                "assert 'context_comparisons.study' not in sys.modules; "
                "assert 'context_comparisons.native' not in sys.modules")
        subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True, text=True)

    def test_transfer_diagnostics_expose_collapse_and_empty_positive_sets(self):
        result = factor_summary(np.zeros(4), np.array([0., 1., 0., 1.]), np.array([True, True, False, False]), .5)
        self.assertTrue(result["native_constant"])
        self.assertEqual(result["native_active_count"], 0)
        self.assertEqual(result["query_activation_outside_rule"], .5)
        self.assertEqual(result["positive_set_jaccard"], 0.)
        self.assertFalse(result["correlation_supported"])
        empty = factor_summary(np.zeros(2), np.zeros(2), np.array([False, False]), .5)
        self.assertTrue(np.isnan(empty["positive_set_jaccard"]))
        relative = reconstruction_summary(np.array([1., 3.]), np.array([2., 6.]))
        self.assertEqual(relative["relative_reconstruction_error"], .5)
        self.assertEqual(relative["standardized_input_energy"], 4.)

    def test_resume_validates_current_identities(self):
        recorded = {"sample_size": 500, "draws": 4, "patient_table_index": "old"}
        with patch.object(provenance, "current", return_value={**recorded, "patient_table_index": "changed"}):
            with self.assertRaisesRegex(ValueError, "patient_table_index"):
                study.signature({}, {"provenance": recorded})

    def test_native_cache_checks_shape_finiteness_and_installed_implementation(self):
        reference = {"context_rows": np.array([1, 2]), "build": "last_3", "reference": 2010,
                     "predictor": {"context_identity": "context", "checkpoint_sha256": "model"},
                     "population": {"fingerprint": "population"}, "origin": 2008, "seed": 42}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(native.predictor, "restore", return_value={}), \
             patch.object(native, "ordered_states", return_value=np.ones((2, 192))), \
             patch.object(native, "implementation_identity", return_value={"library": "first"}) as implementation:
            states = native.extract(reference, directory)
            target = Path(directory) / "native_last_3_2010.npz"
            np.testing.assert_array_equal(native.extract(reference, directory), states)
            implementation.return_value = {"library": "second"}
            with self.assertRaisesRegex(ValueError, "source context"):
                native.extract(reference, directory)
            implementation.return_value = {"library": "first"}
            with np.load(target) as saved:
                payload = {key: saved[key].copy() for key in saved.files}
            for invalid in (np.ones((1, 192)), np.full((2, 192), np.nan)):
                np.savez(target, **{**payload, "states": invalid})
                with self.assertRaisesRegex(ValueError, "shape or nonfinite"):
                    native.extract(reference, directory)

    def test_report_excludes_full_context_fallback_from_common_size_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            rows = []
            for method, form, size, tuned in (("original", "original", 1000, .3),
                    ("uniform_sampling", "uniform", 500, .4),
                    ("native_context_concepts_hard_sampling", "original", 1000, np.nan)):
                for patient, label, probability in (("a", 0, .1), ("b", 1, .8)):
                    rows.append({"build": "last_3", "reference": 2010, "year": 2011,
                                 "update_year": 2010, "patient_id": patient, "sampling_seed": 42,
                                 "label": label, "probability": probability, "method": method,
                                 "threshold": .5, "tuned_threshold": tuned, "reference_mortality": .2,
                                 "effective_form": form, "context_size": size,
                                 "fallback": method.startswith("native"), "common_size_eligible": form == "uniform"})
            pd.DataFrame(rows).to_csv(output / "predictions_last_3_2010_2010.csv.gz", index=False)
            reporting.summarize(output, repetitions=2)
            metrics = pd.read_csv(output / "intervention_metrics.csv.gz")
            self.assertEqual(set(metrics["threshold_type"]), {"threshold"})
            intervals = pd.read_csv(output / "paired_intervals.csv.gz")
            wrong = ((intervals["method"] == "native_context_concepts_hard_sampling") &
                     (intervals["control"] == "uniform_sampling"))
            self.assertFalse(wrong.any())

    def test_choice_coverage_separates_changed_and_unchanged_choices(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            decisions = []
            for changed, evaluation in ((True, "new test inference required"), (False, "cached")):
                decisions.append({"objective": "average_precision", "different_from_death_f1": changed,
                                  "evaluation": evaluation, "choice": {"form": "uniform", "strength": 0.}})
            (output / "metric_choices_example.json").write_text(json.dumps({"decisions": decisions,
                                                                           "new_model_inferences": 0}))
            reporting.summarize_choices(output)
            coverage = pd.read_csv(output / "objective_coverage.csv")
            self.assertEqual(int(coverage["changed_choices"].iloc[0]), 1)
            self.assertEqual(int(coverage["changed_choices_evaluated"].iloc[0]), 0)
            self.assertEqual(int(coverage["unchanged_choices_evaluated"].iloc[0]), 1)


if __name__ == "__main__":
    unittest.main()
