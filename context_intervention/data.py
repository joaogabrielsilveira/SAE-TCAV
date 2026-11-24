"""Select the frozen study rows and construct comparable representations."""

import numpy as np

from context_intervention.sampling import patient_partition


DEVELOPMENT_ROLES = ("sae_discovery", "rule_discovery", "cav_fit", "concept_selection")


def development_rows(reference, last_year, partition):
    """Rows whose outcomes are available at an update, split by patient."""
    if partition not in ("development", "validation"):
        raise ValueError("Unknown development partition")
    population, roles = reference["population"], reference["roles"]
    years = population["year"]
    if not reference["reference"] <= last_year <= 2013:
        raise ValueError("Update year must be between the reference and 2013")
    allowed_roles = np.isin(roles["row_role"], DEVELOPMENT_ROLES)
    allowed_roles |= (roles["row_role"] == "fitting") & roles["is_entrant"]
    eligible = ((years >= reference["reference"]) & (years <= last_year) & allowed_roles &
                np.isin(population["y"], [0, 1]))
    parts = patient_partition(population["patient_id"])
    rows = np.flatnonzero(eligible & (parts == partition))
    if np.intersect1d(rows, reference["context_rows"]).size:
        raise ValueError("Development query overlaps the original context by row")
    return rows


def evaluation_rows(reference, year):
    population, roles = reference["population"], reference["roles"]
    if not reference["reference"] < year <= 2014:
        raise ValueError("Evaluation year must follow the reference and be at most 2014")
    return np.flatnonzero((population["year"] == year) & (roles["row_role"] == "evaluation") &
                          np.isin(population["y"], [0, 1]))


def cached_predictor_rows(reference, rows):
    """Return cached probabilities and embeddings after exact row identity checks."""
    rows = np.asarray(rows, dtype=int)
    predictor = reference["predictor"]
    indices = np.asarray(predictor["row_index"])
    if np.any(np.diff(indices) <= 0):
        raise ValueError("Predictor row index must be strictly sorted")
    positions = np.searchsorted(indices, rows)
    if np.any(positions >= len(indices)) or not np.array_equal(indices[positions], rows):
        raise ValueError("Some requested rows are missing from the frozen predictions")
    return predictor["probabilities"][positions, 1], predictor["embeddings"][positions]


def standardized(values, reference_values):
    """Apply fixed reference statistics; constant columns contribute zero."""
    values, reference_values = np.asarray(values, dtype=float), np.asarray(reference_values, dtype=float)
    if values.ndim != 2 or reference_values.ndim != 2 or values.shape[1] != reference_values.shape[1]:
        raise ValueError("Representation matrices must have the same columns")
    if not len(reference_values) or not np.isfinite(values).all() or not np.isfinite(reference_values).all():
        raise ValueError("Representations and reference statistics must be finite and nonempty")
    mean, std = reference_values.mean(axis=0), reference_values.std(axis=0)
    return np.divide(values - mean, std, out=np.zeros_like(values), where=std > 0)


def factor_activations(reference, embeddings, concept_records, device):
    """Encode query-position embeddings with frozen dictionaries and select medoids."""
    import concept_subgroup_drift_pilot as pilot

    dictionaries = pilot.load_reference_dictionaries(reference["cache"], reference["reference"], device)
    seeds = {record["seed"] for record in concept_records}
    codes = {seed: pilot.encode_with_reference_sae(dictionaries[seed], embeddings, device, 256)
             for seed in seeds}
    return np.column_stack([codes[record["seed"]][:, record["factor"]] for record in concept_records])
