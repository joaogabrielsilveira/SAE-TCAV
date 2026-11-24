"""Read existing inference caches and calculate only genuinely missing predictions."""

import hashlib
from pathlib import Path

import numpy as np

from context_intervention import experiment, predictor


class MissingPredictions(RuntimeError):
    """An optional comparison would require additional model inference."""


def cache_identity(reference, context_rows, query_rows):
    """Match the original runner's key without editing its source or caches."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(context_rows, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(query_rows, dtype=np.int64).tobytes())
    for value in (reference["predictor"]["context_identity"], reference["predictor"]["checkpoint_sha256"],
                  reference["population"]["fingerprint"], reference["origin"], reference["seed"]):
        digest.update(str(value).encode())
    for path in (Path(experiment.__file__), Path(predictor.__file__),
                 Path("concept_subgroup_drift_pilot.py"), Path("tabpfn_model.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


class PredictionStore:
    def __init__(self, reference, original_directory, directory, allow_inference=False, supersets=()):
        self.reference = reference
        self.original_directory = Path(original_directory)
        self.directory = Path(directory)
        self.allow_inference = allow_inference
        self.supersets = [np.asarray(rows, dtype=int) for rows in supersets]
        self.hits = 0
        self.new_predictions = 0

    def __call__(self, context_rows, query_rows):
        query_rows = np.asarray(query_rows, dtype=int)
        possibilities = [query_rows]
        possibilities += [rows for rows in self.supersets if len(rows) > len(query_rows) and np.isin(query_rows, rows).all()]
        for rows in possibilities:
            identity = cache_identity(self.reference, context_rows, rows)
            for directory in (self.original_directory, self.directory):
                path = directory / f"{identity}.npz"
                if path.exists():
                    with np.load(path, allow_pickle=False) as stored:
                        if (str(stored["identity"]) != identity or stored["probabilities"].shape != (len(rows),)
                                or not np.isfinite(stored["probabilities"]).all()):
                            raise ValueError("Cached prediction identity or shape mismatch")
                        lookup = {int(row): index for index, row in enumerate(rows)}
                        positions = np.array([lookup[int(row)] for row in query_rows], dtype=int)
                        self.hits += 1
                        return stored["probabilities"][positions].copy()
        if not self.allow_inference:
            raise MissingPredictions("Selected option needs new inference; cached-only comparison retained its decision")
        probabilities, _ = experiment.score_context(self.reference, context_rows, query_rows, self.directory)
        self.new_predictions += 1
        return probabilities
