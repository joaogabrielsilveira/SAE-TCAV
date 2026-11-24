"""Checks of chronological role and patient isolation."""

import unittest

import numpy as np

from context_intervention.data import development_rows, evaluation_rows
from context_intervention.sampling import patient_partition


class DataTests(unittest.TestCase):
    def test_years_and_roles_cannot_enter_early(self):
        ids = np.array(["a", "b", "c", "a", "d", "e", "f"])
        ref = {"reference": 2010, "context_rows": np.array([0]),
               "population": {"year": np.array([2010, 2010, 2010, 2011, 2011, 2012, 2015]),
                              "patient_id": ids, "y": np.array([0, 1, 0, 0, 1, 0, 1]),
                              "post_death": np.zeros(7, dtype=bool)},
               "roles": {"row_role": np.array(["tabpfn_context", "rule_discovery", "evaluation", "fitting",
                                                "fitting", "sae_discovery", "fitting"]),
                         "is_entrant": np.array([0, 0, 0, 1, 1, 0, 1], dtype=bool)}}
        first = np.concatenate([development_rows(ref, 2010, "development"),
                                development_rows(ref, 2010, "validation")])
        self.assertEqual(set(first), {1})
        later = np.concatenate([development_rows(ref, 2011, "development"),
                                development_rows(ref, 2011, "validation")])
        self.assertEqual(set(later), {1, 3, 4})
        self.assertEqual(patient_partition(ids)[0], patient_partition(ids)[3])
        self.assertEqual(len(evaluation_rows(ref, 2011)), 0)
        with self.assertRaises(ValueError):
            evaluation_rows(ref, 2015)


if __name__ == "__main__":
    unittest.main()
