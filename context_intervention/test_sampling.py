"""Checks of the sampling invariants used by the planned comparisons."""

import unittest

import numpy as np

from context_intervention import sampling


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.years = np.repeat([2010, 2011], 50)
        self.labels = np.tile(np.repeat([0, 1], [40, 10]), 2)

    def test_stratified_sample_is_reproducible_and_without_replacement(self):
        first = sampling.sample_context(self.years, self.labels, 40, 42)
        second = sampling.sample_context(self.years, self.labels, 40, 42)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(first)), 40)
        self.assertEqual(sampling.quotas(sampling.strata(self.years[first], self.labels[first]), 40),
                         sampling.quotas(sampling.strata(self.years, self.labels), 40))

    def test_removal_and_random_control_preserve_stratum_counts(self):
        priorities = np.arange(100, dtype=float)
        retained = sampling.select_removals(self.years, self.labels, priorities, .5, 40)
        random = sampling.random_matched_removal(self.years, self.labels, ~retained, 1)
        for year, label in set(sampling.strata(self.years, self.labels)):
            rows = (self.years == year) & (self.labels == label)
            self.assertEqual(np.sum(rows & ~retained), np.sum(rows & random))
        sample = sampling.sample_context(self.years, self.labels, 40, 2, eligible=retained)
        self.assertTrue(retained[sample].all())

    def test_weights_and_patient_partition(self):
        np.testing.assert_array_equal(sampling.graded_weights(np.zeros(5), np.log(2)), np.ones(5))
        weights = sampling.graded_weights(np.array([-1., 0., 1.]), np.log(2))
        np.testing.assert_allclose(weights, [2., 1., .5])
        parts = sampling.patient_partition(["a", "b", "a"])
        self.assertEqual(parts[0], parts[2])

    def test_neighborhood_and_missing_support(self):
        vectors = np.arange(200, dtype=float).reshape(100, 2)
        chosen = sampling.neighborhood(vectors, vectors[[3, 40]], self.years, self.labels, 20)
        self.assertEqual(len(np.unique(chosen)), 20)
        with self.assertRaises(ValueError):
            sampling.sample_context(self.years, self.labels, 40, 1, eligible=np.zeros(100, dtype=bool))

    def test_exclusion_only_replaces_affected_rows(self):
        draw = sampling.sample_context(self.years, self.labels, 40, 42)
        excluded = np.zeros(100, dtype=bool)
        excluded[draw[::5]] = True
        replacement = sampling.replace_neighborhood(self.years, self.labels, draw, excluded, 9)
        self.assertFalse(excluded[replacement].any())
        self.assertTrue(set(draw[~excluded[draw]]).issubset(set(replacement)))
        self.assertEqual(len(np.unique(replacement)), len(draw))
        self.assertEqual(sampling.quotas(sampling.strata(self.years[draw], self.labels[draw]), len(draw)),
                         sampling.quotas(sampling.strata(self.years[replacement], self.labels[replacement]), len(draw)))

    def test_group_effects_are_averaged_only_over_observed_neighborhoods(self):
        priorities = sampling.exclusion_priorities(4, [[0, 1], [1, 2]], [.2, -.1])
        np.testing.assert_allclose(priorities, [.2, .05, -.1, 0.])

    def test_soft_weights_change_exposure_without_dropping_rows(self):
        weights = sampling.graded_weights(np.r_[np.ones(50), -np.ones(50)], np.log(4))
        self.assertTrue((weights > 0).all())
        self.assertLess(weights[0], weights[-1])
        selected = sampling.sample_context(self.years, self.labels, 40, 42, weights=weights)
        self.assertEqual(len(np.unique(selected)), 40)


if __name__ == "__main__":
    unittest.main()
