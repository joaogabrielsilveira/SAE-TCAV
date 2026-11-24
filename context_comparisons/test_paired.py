"""Checks for paired sampling, similarity-ordered removal and the stable patient split."""

import unittest

import numpy as np

from context_comparisons import paired
from context_intervention import sampling


def context(n=5520, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice([2006, 2007, 2008], size=n), (rng.random(n) < .1).astype(int)


class PairedSamplingTests(unittest.TestCase):
    def test_uniform_draw_is_reproducible_and_keeps_original_quotas(self):
        years, labels = context()
        first = paired.sample_context(years, labels, 500, 7)
        np.testing.assert_array_equal(first, paired.sample_context(years, labels, 500, 7))
        self.assertEqual(len(np.unique(first)), 500)
        expected = sampling.quotas(sampling.strata(years, labels), 500)
        actual = sampling.quotas(sampling.strata(years[first], labels[first]), 500)
        self.assertEqual(expected, actual)

    def test_excluding_rows_changes_only_those_rows(self):
        years, labels = context()
        for seed in range(20):
            uniform = paired.sample_context(years, labels, 500, seed)
            eligible = np.ones(len(years), dtype=bool)
            eligible[np.random.default_rng(seed).choice(len(years), 10, replace=False)] = False
            hard = paired.sample_context(years, labels, 500, seed, eligible)
            removed_from_draw = int((~eligible[uniform]).sum())
            self.assertEqual(len(np.intersect1d(uniform, hard)), 500 - removed_from_draw)
            self.assertTrue(eligible[hard].all())
            # The original study's sampler redraws the strata: overlap is far lower.
            original = sampling.sample_context(years, labels, 500, seed)
            original_hard = sampling.sample_context(years, labels, 500, seed, eligible)
            self.assertLess(len(np.intersect1d(original, original_hard)), 400)

    def test_equal_weights_reproduce_uniform_and_small_weights_are_drawn_less(self):
        years, labels = context(1000)
        np.testing.assert_array_equal(paired.sample_context(years, labels, 100, 3),
                                      paired.sample_context(years, labels, 100, 3, weights=np.full(1000, 2.)))
        weights = np.ones(1000)
        weights[:200] = .25
        counts = np.zeros(1000)
        for seed in range(400):
            counts[paired.sample_context(years, labels, 100, seed, weights=weights)] += 1
        self.assertLess(counts[:200].mean(), .5 * counts[200:].mean())
        # Graded draws stay close to the uniform draw with the same keys.
        overlap = [len(np.intersect1d(paired.sample_context(years, labels, 100, seed),
                                      paired.sample_context(years, labels, 100, seed, weights=weights)))
                   for seed in range(50)]
        self.assertGreater(np.mean(overlap), 60)

    def test_neighborhood_exclusion_keeps_every_uniform_row_outside_it(self):
        years, labels = context(2000)
        uniform = paired.sample_context(years, labels, 200, 11)
        eligible = np.ones(2000, dtype=bool)
        eligible[:300] = False
        changed = paired.sample_context(years, labels, 200, 11, eligible)
        self.assertTrue(set(uniform[uniform >= 300]).issubset(changed))


class RemovalOrderTests(unittest.TestCase):
    def test_counts_match_the_original_and_ties_follow_similarity(self):
        years, labels = np.array([2008] * 10), np.array([0] * 8 + [1] * 2)
        priorities = np.array([.1] * 8 + [0., 0.])
        closeness = np.array([.1, .2, .9, .8, .3, .4, .5, .6, 0., 0.])
        retained = paired.select_removals(years, labels, priorities, closeness, .25, 5)
        original = sampling.select_removals(years, labels, priorities, .25, 5)
        self.assertEqual((~retained).sum(), (~original).sum())
        self.assertEqual(set(np.flatnonzero(~retained)), {2, 3})
        self.assertEqual(set(np.flatnonzero(~original)), {0, 1})
        # A higher priority still wins over higher similarity.
        priorities[0] = .2
        retained = paired.select_removals(years, labels, priorities, closeness, .25, 5)
        self.assertEqual(set(np.flatnonzero(~retained)), {0, 2})


class SplitTests(unittest.TestCase):
    def reference(self):
        patients = np.tile(np.arange(100).astype(str), 2).astype(object)
        return {"reference": 2008, "context_rows": np.array([], dtype=int),
                "population": {"patient_id": patients, "year": np.repeat([2008, 2009], 100),
                               "y": np.zeros(200, dtype=int)},
                "roles": {"row_role": np.full(200, "sae_discovery"), "is_entrant": np.zeros(200, bool)}}

    def test_future_outcomes_cannot_change_earlier_assignments(self):
        reference = self.reference()
        before = paired.development_rows(reference, 2008, "development")
        for seed in range(10):
            reference["population"]["y"][100:] = np.random.default_rng(seed).integers(0, 2, 100)
            np.testing.assert_array_equal(before, paired.development_rows(reference, 2008, "development"))

    def test_split_matches_original_and_keeps_patients_together_across_years(self):
        from context_intervention import data

        reference = self.reference()
        for year in (2008, 2009):
            for partition in ("development", "validation"):
                np.testing.assert_array_equal(paired.development_rows(reference, year, partition),
                                              data.development_rows(reference, year, partition))
        patients = reference["population"]["patient_id"]
        development = paired.development_rows(reference, 2009, "development")
        validation = paired.development_rows(reference, 2009, "validation")
        self.assertFalse(set(patients[development]) & set(patients[validation]))

    def test_future_entrants_cannot_change_earlier_assignments(self):
        reference = self.reference()
        before = paired.development_rows(reference, 2008, "development")
        reference["population"]["patient_id"][100:] = [f"future-{i}" for i in range(100)]
        np.testing.assert_array_equal(before, paired.development_rows(reference, 2008, "development"))


if __name__ == "__main__":
    unittest.main()
