"""Small, outcome-oriented checks for the frozen-baseline comparison."""

import unittest

import numpy as np
import pandas as pd

from context_intervention.analysis import performance, prediction_groups
from context_intervention.evaluation import choose_threshold, paired_patient_bootstrap, prediction_transitions


class EvaluationTests(unittest.TestCase):
    def test_metrics_and_prediction_transitions(self):
        labels = np.array([1, 1, 0, 0])
        baseline = np.array([.8, .2, .8, .2])
        modified = np.array([.8, .8, .2, .8])
        result = performance(labels, baseline, .5, .25)
        self.assertAlmostEqual(result["death_f1"], .5)
        self.assertEqual(prediction_transitions(labels, baseline, modified, .5)["corrected"], 2)
        self.assertEqual(prediction_transitions(labels, baseline, modified, .5)["damaged"], 1)
        np.testing.assert_array_equal(prediction_groups(labels, baseline, .5),
                                      ["true_positive", "false_negative", "false_positive", "true_negative"])

    def test_threshold_uses_only_supplied_validation_outcomes(self):
        labels, probabilities = [1, 1, 0, 0], [.8, .6, .4, .1]
        threshold = choose_threshold(labels, probabilities, .5, .25)
        self.assertEqual(threshold, .5)
        self.assertEqual(choose_threshold([0], [.2], .5, .25), .5)

    def test_paired_bootstrap_identical_probabilities_have_zero_difference(self):
        frame = pd.DataFrame({"patient_id": ["a", "b", "a", "c"], "reference": [2008]*4,
                              "year": [2009, 2009, 2010, 2010], "label": [1, 0, 1, 0],
                              "threshold": [.5]*4, "reference_mortality": [.25]*4,
                              "baseline_probability": [.8, .2, .7, .1], "probability": [.8, .2, .7, .1]})
        result = paired_patient_bootstrap(frame, repetitions=10, seed=42)
        for value in result.values():
            self.assertEqual(value["difference"], 0.)
            self.assertEqual(value["lower"], 0.)
            self.assertEqual(value["upper"], 0.)


if __name__ == "__main__":
    unittest.main()
