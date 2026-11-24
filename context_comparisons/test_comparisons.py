"""Checks for temporal filtering, objective selection, cache reuse and native coordinates."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from context_comparisons import choices, methods
from context_comparisons.cache import MissingPredictions, PredictionStore


class ComparisonTests(unittest.TestCase):
    def test_objective_changes_only_the_ranking_of_existing_options(self):
        records = [{"form": "original", "strength": 0., "death_f1": .4, "brier_skill": .1, "average_precision": .3},
                   {"form": "direct", "strength": .25, "death_f1": .35, "brier_skill": .2, "average_precision": .32},
                   {"form": "direct", "strength": .5, "death_f1": .3, "brier_skill": .15, "average_precision": .4}]
        self.assertEqual(choices.choose(records, "death_f1")["strength"], 0.)
        self.assertEqual(choices.choose(records, "brier_skill")["strength"], .25)
        self.assertEqual(choices.choose(records, "average_precision")["strength"], .5)
        tie = [dict(records[0]), {**records[0], "form": "direct", "strength": .5}]
        self.assertEqual(choices.choose(tie, "death_f1")["strength"], 0.)

    def test_cached_superset_preserves_requested_order_without_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            rows = np.array([10, 11, 12, 13])
            with patch("context_comparisons.cache.cache_identity", return_value="cache"):
                np.savez(path / "cache.npz", identity="cache", probabilities=np.array([.1, .2, .3, .4]), device="cpu")
                store = PredictionStore({}, path, path / "new", supersets=[rows])
                # Exact-key shape validation is intentionally strict. Request
                # identity differs for the subset, so simulate both keys.
                def key(reference, context, queries):
                    return "cache" if len(queries) == 4 else "missing"
                with patch("context_comparisons.cache.cache_identity", side_effect=key):
                    result = store([1, 2], [13, 11])
                np.testing.assert_allclose(result, [.4, .2])
                self.assertEqual(store.new_predictions, 0)

    def test_missing_optional_prediction_never_launches_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            store = PredictionStore({}, directory, directory, allow_inference=False)
            with patch("context_comparisons.cache.cache_identity", return_value="absent"):
                with self.assertRaises(MissingPredictions):
                    store([1, 2], [3])

    def test_latest_year_masks_both_candidate_and_validation_rows(self):
        from context_comparisons import study

        reference = {"build": "last_3", "reference": 2010, "context_rows": np.array([0, 1]),
                     "population": {"year": np.array([2010, 2010, 2010, 2011, 2010, 2011, 2012]),
                                    "y": np.array([0, 1, 1, 0, 0, 1, 0])}}
        def rows(reference, year, part):
            return np.array([2, 3]) if part == "development" else np.array([4, 5])
        with patch.object(study.original_study, "reference_info", return_value=({"death_threshold": {"last_3": .5}}, [])), \
             patch.object(study.measures, "load_build", return_value={"reference_mortality": .2}), \
             patch.object(study.paired, "development_rows", side_effect=rows), \
             patch.object(study.data, "evaluation_rows", return_value=np.array([6])), \
             patch.object(study.data, "cached_predictor_rows", return_value=(np.array([.2]), None)), \
             patch.object(study.paired, "predictions", return_value=np.array([[.2], [.2], [.2]])), \
             patch.object(study, "frame", return_value=None):
            frames, result, priorities = study.evaluate_variant(reference, 2011, "features",
                (np.ones((2, 2)), np.ones((2, 2))), "latest_year", "latest_year",
                {"provenance": {"sample_size": 1, "draws": 1}})
        self.assertEqual(result["counts"]["candidate_rows"], 1)
        self.assertEqual(result["counts"]["validation_rows"], 1)
        self.assertTrue(all(value["winner"]["form"] == "original" for value in result["choices"].values()))

    def test_latest_year_effects_and_validation_use_only_the_latest_year(self):
        from context_comparisons import study

        reference = {"build": "last_3", "reference": 2010, "context_rows": np.array([0, 1]),
                     "population": {"year": np.array([2010, 2010, 2010, 2011, 2011, 2010, 2011, 2011, 2012]),
                                    "y": np.array([0, 1, 1, 1, 0, 0, 0, 1, 0])}}
        observed = {"effects": [], "queries": []}
        def rows(reference, year, part):
            return np.array([2, 3, 4]) if part == "development" else np.array([5, 6, 7])
        def effects(reference, neighborhoods, queries, *args):
            observed["effects"].append(queries.tolist())
            return np.zeros(2), []
        def predictions(reference, priorities, closeness, queries, *args):
            observed["queries"].append(queries.tolist())
            return np.full((3, len(queries)), .2)
        with patch.object(study.original_study, "reference_info", return_value=({"death_threshold": {"last_3": .5}}, [])), \
             patch.object(study.measures, "load_build", return_value={"reference_mortality": .2}), \
             patch.object(study.paired, "development_rows", side_effect=rows), \
             patch.object(study.data, "evaluation_rows", return_value=np.array([8])), \
             patch.object(study.data, "cached_predictor_rows", return_value=(np.array([.7, .2]), np.ones((2, 192)))), \
             patch.object(study.experiment, "make_neighborhoods", return_value=([], [])), \
             patch.object(study, "retrieval_diagnostics", return_value={"any_discriminating_stratum": True}), \
             patch.object(study.paired, "neighborhood_priorities", side_effect=effects), \
             patch.object(study.paired, "predictions", side_effect=predictions), \
             patch.object(study, "frame", return_value=None):
            study.evaluate_variant(reference, 2011, "features", (np.eye(2), np.ones((3, 2))),
                "latest_year", "latest_year", {"provenance": {"sample_size": 1, "draws": 1}})
        self.assertEqual(observed["effects"], [[3, 4]])
        validation = [queries for queries in observed["queries"] if queries != [8]]
        self.assertEqual(validation, [[6, 7]] * 9)


if __name__ == "__main__":
    unittest.main()
