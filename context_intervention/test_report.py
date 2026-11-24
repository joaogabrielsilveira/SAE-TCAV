"""Check pairing, temporal aggregation and report output on small examples."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from context_intervention.report import summarize


class ReportTests(unittest.TestCase):
    def test_report_pairs_each_method_with_the_same_patient_and_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = []
            for method in ("original", "uniform_sampling", "features_hard_sampling", "features_graded_sampling"):
                for seed in (42, 135, 215):
                    for patient, label, probability in (("a", 1, .7), ("b", 0, .2)):
                        rows.append({"build": "all_history", "reference": 2008, "year": 2009,
                                     "patient_id": patient, "sampling_seed": seed, "method": method,
                                     "label": label, "probability": probability, "threshold": .5,
                                     "tuned_threshold": .5, "reference_mortality": .2})
            path = Path(directory) / "predictions_all_history_2008_2008.csv.gz"
            pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
            result = summarize(directory, bootstrap_repetitions=3)
            self.assertEqual(result["completed_reference_year_cells"], 1)
            intervals = pd.read_csv(Path(directory) / "paired_intervals.csv.gz")
            control = intervals[(intervals.method == "features_graded_sampling") &
                                (intervals.control == "features_hard_sampling")]
            self.assertTrue((control.difference == 0).all())
            self.assertGreater(len(control), 0)


if __name__ == "__main__":
    unittest.main()
